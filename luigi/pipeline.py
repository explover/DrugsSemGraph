import luigi

import pickle as pkl
import requests, json
import pandas as pd
import func #the module containing all functions
from tqdm import tqdm
import matplotlib.pyplot as plt


#INPUT = './data/ncomms10331-s2.xls'
#INPUT_KW = {'skiprows': 1,
#           'skip_footer': 1}

class RetrieveIDs(luigi.Task):
    '''This Task retrieves IDs of diseases and drugs and writes them to a file.
    Index of the DataFrame remains unchanged'''
    data_path = luigi.Parameter(description="Path to the data.")
    
    def output(self):
        return [luigi.LocalTarget(f"{self.data_path}/id_table_pos.hdf"),
                luigi.LocalTarget(f"{self.data_path}/id_table_neg.hdf")]
    
    
    def run(self):
        dsets = ["neg"]#["pos", "neg"]
        for dset in dsets:
            table = pd.read_csv(f"{self.data_path}/{dset}.csv")
            print(table)
            disease_ids = table.apply(func.return_disease_ids, axis=1)
            drug_ids = table.apply(func.return_drug_ids, axis=1)
            combined_ids = pd.DataFrame(index=disease_ids.index)
            combined_ids["disease_ids"] = disease_ids
            combined_ids["drug_ids"] = drug_ids
            dis_ok = func.list_not_empty(combined_ids["disease_ids"])
            drug_ok = func.list_not_empty(combined_ids["drug_ids"])
            combined_ids = combined_ids[dis_ok & drug_ok]
        
            with open(f"{self.data_path}/{dset}_num.pkl", "wb") as num_file:
                pkl.dump(len(combined_ids), num_file)
            combined_ids.to_hdf(f"{self.data_path}/id_table_{dset}.hdf", 'w', mode='w')
        

class GetNegativeSet(luigi.Task):
    '''This Task produces certain number of drug-disease negative sets.'''
    table_number = luigi.IntParameter(default=20) 
    def requires(self):
        return RetrieveIDs()
    
    def output(self):
        out_list = [luigi.LocalTarget("id_table_neg_{}.hdf".format(table_id)) 
                                  for table_id in range(self.table_number)]
        out_list.append(luigi.LocalTarget("table_number.pkl"))
        return out_list
        
    def run(self):
        with open("table_number.pkl", "wb") as number_file:
            pkl.dump(self.table_number, number_file)
        id_table = pd.read_hdf('id_table.hdf', 'w')
        func.randomize_drugs_diseases(self.table_number, id_table)
        
class FindDirect(luigi.Task):
    '''This Task finds direct relations between lists of IDs in the rows of id_table.''' 
    data_path = luigi.Parameter(description="Path to the data.")

    def requires(self):
        return [RetrieveIDs(data_path=self.data_path)]#, GetNegativeSet()]
    
    def output(self):
        return [luigi.LocalTarget(f"{self.data_path}/direct_{dset}.hdf") 
                for dset in ["pos", "neg"]] 
    
    def run(self):
        dsets = ["pos", "neg"]
        for dset in tqdm(dsets):
            id_table = pd.read_hdf(f"{self.data_path}/id_table_{dset}.hdf", 'w')
            direct_relations = id_table.apply(func.find_direct_relations, axis=1) 
            direct_relations.to_hdf(f"{self.data_path}/direct_{dset}.hdf", 'w', mode='w')
         
class RetrieveDirectRelationInfo(luigi.Task):
    '''This Task retrieves important information on direct relations.'''
    data_path = luigi.Parameter(description="Path to the data.")
    
    def requires(self):
        return FindDirect(data_path=self.data_path)
    
    def output(self):
        return [luigi.LocalTarget(f"{self.data_path}/direct_short_{dset}.hdf") 
                for dset in ["pos", "neg"]] 
    
    def run(self): 
        dsets = ["pos", "neg"]
        for dset in tqdm(dsets):
            direct_relations = pd.read_hdf(f"{self.data_path}/direct_{dset}.hdf", 'w')
            dir_rel_info = direct_relations.apply(func.pull_direct_relations_info) 
            dir_rel_info.to_hdf(f"{self.data_path}/direct_short_{dset}.hdf", 'w', mode='w')
    
class FindIndirect(luigi.Task):
    '''This Task finds indirect paths between list of IDS in the rows of positive id_table.'''
    data_path = luigi.Parameter(description="Path to the data.")
    #pos_sample_number = luigi.IntParameter(description="Number of objects in positive dataset")
    #neg_sample_number = luigi.IntParameter(description="Number of objects in negative dataset")

    def requires(self):
        return [RetrieveIDs(data_path=self.data_path)]#, GetNegativeSet()]
    
    def output(self):
        num_dict = dict()
        for dset in ["pos", "neg"]:
            with open(f"{self.data_path}/{dset}_num.pkl", "rb") as num_file:
                num = pkl.load(num_file)
            num_dict[dset] = num
        return [luigi.LocalTarget(f"{self.data_path}/indirect_jsons/{dset}_{row_number}.json") 
                for dset in ["pos", "neg"]
                for row_number in range(num_dict[dset])]
    
    def run(self):
        dsets = ["pos", "neg"]
        for dset in dsets:
            id_table = pd.read_hdf(f"{self.data_path}/id_table_{dset}.hdf", 'w')
            for row_number, (_, row) in tqdm(enumerate(id_table.iterrows())):
                func.get_indirect(row, dset, row_number, self.data_path)

class GetFeatureTables(luigi.Task):
    '''This Task forms feature tables with counts and diversities.'''
    data_path = luigi.Parameter(description="Path to the data.")

    def requires(self):
        return [FindDirect(data_path=self.data_path),
                FindIndirect(data_path=self.data_path)]

    def output(self):
        return [luigi.LocalTarget(f"{self.data_path}/feature_table_{dset}.csv")
                for dset in ["pos", "neg"]]

    def run(self):
        dsets = ["pos", "neg"]
        for dset in tqdm(dsets):
            id_table = pd.read_hdf(f"{self.data_path}/id_table_{dset}.hdf", 'w')
            id_table.index = range(len(id_table))
            semgroups_filename = f"{self.data_path}/SemGroups.txt"
            feature_table = func.get_feature_table(id_table, dset, 
                                                   semgroups_filename,
                                                   self.data_path)
            feature_table.to_csv(f"{self.data_path}/feature_table_{dset}.csv",
                                 index=False)
        
class PlotDistributions(luigi.Task):
    '''This Task plots distributions of normalized counts within
    semantic categories.'''
    protocol = luigi.Parameter(default="semanticCategory")
    #with open("table_number.pkl", "rb") as number_file:
    #    table_number = pkl.load(number_file)
    table_number = 10

    #def requires(self):
    #    return FindIndirect()

    def output(self):
        self.random_counter, self.nonrandom_counter, self.cats = self.get_all()
        return [luigi.LocalTarget("./{}_density_pictures/{}.png".format(self.protocol, cat)) 
                for cat in self.cats]

    def run(self):
        for cat in self.cats:
            #func.plot_both(cat, self.nonrandom_counter, self.random_table_counters)
            direct_real = self.nonrandom_counter[0]
            nondirect_real = self.nonrandom_counter[1]
            direct_random = self.random_counter[0]
            nondirect_random = self.random_counter[1]
            plt.figure()

            patch1 = func.plot_dens(cat, direct_real, color="red", label="direct real", ls="solid") 
            #patch2 = func.plot_dens(cat, nondirect_real, color="red", label="nondirect real", ls="dashed") 
            patch3 = func.plot_dens(cat, direct_random, color="blue", label="direct random", ls="solid") 
            patch4 = func.plot_dens(cat, nondirect_random, color="blue", label="nondirect random", ls="dashed") 

            plt.legend(handles = [patch1, patch3, patch4])
            plt.title(cat)
            plt.savefig("./{}_density_pictures/{}.png".format(self.protocol, cat), format="png")
            plt.close()

    def get_all(self):
        id_table = pd.read_hdf('id_table.hdf', 'w') 
        id_table["direct"] = pd.read_hdf('direct_short_real.hdf', 'w')
        random_table_counters = func.get_counters(self.table_number, protocol=self.protocol)
        nonrandom_counter = list(func.get_sem_cat_counter_list(id_table, "real", protocol=self.protocol))
        random_counter = [list(), list()]
        for direct, nondirect in random_table_counters:
            random_counter[0] = random_counter[0] + direct
            random_counter[1] = random_counter[1] + nondirect

        all_stuff = [random_counter[0] + nonrandom_counter[0],
                     random_counter[1] + nonrandom_counter[1]]
        cats = set()
        for direct_counter, nondirect_counter in zip(*all_stuff):
                cats = cats | set(direct_counter.keys()) | set(nondirect_counter.keys())
        return random_counter, nonrandom_counter, cats
        
class PlotDiversity(luigi.Task):
    '''This Task plots distributions of normalized diversity.'''
    protocol = "diversity"
    table_number = 10
    #def requires(self):
    #    return FindIndirect()

    def output(self):
        return luigi.LocalTarget("./pictures/diversity.png")

    def run(self): 
        id_table = pd.read_hdf('id_table.hdf', 'w') 
        id_table["direct"] = pd.read_hdf('direct_short_real.hdf', 'w')
        random_table_counters = func.get_counters(self.table_number, protocol=self.protocol)
        nonrandom_counter = func.get_sem_cat_counter_list(id_table, "real", protocol=self.protocol)
        random_counter = [list(), list()]
        for direct, nondirect in random_table_counters:
            random_counter[0] = random_counter[0] + direct
            random_counter[1] = random_counter[1] + nondirect
        
        direct_real = nonrandom_counter[0]
        nondirect_real = nonrandom_counter[1]
        direct_random = random_counter[0]
        nondirect_random = random_counter[1]
        plt.figure()

        patch1 = func.plot_dens_diversity(direct_real, color="red", label="direct real", ls="solid") 
        #patch2 = func.plot_dens_diversity(nondirect_real, color="red", label="nondirect real", ls="dashed") 
        patch3 = func.plot_dens_diversity(direct_random, color="blue", label="direct random", ls="solid") 
        patch4 = func.plot_dens_diversity(nondirect_random, color="blue", label="nondirect random", ls="dashed") 

        plt.legend(handles = [patch1, patch3, patch4])
        plt.title("Diversity")
        plt.savefig("./{}_density_pictures/diversity.png".format(self.protocol), format="png")
        plt.close()
        
if __name__ == '__main__':
        luigi.run()
