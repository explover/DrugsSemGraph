import luigi

import pickle as pkl
import requests, json
import pandas as pd
import func #the module containing all functions
from tqdm import tqdm
import matplotlib.pyplot as plt


INPUT = './data/ncomms10331-s2.xls'
INPUT_KW = {'skiprows': 1,
           'skip_footer': 1}

class RetrieveIDs(luigi.Task):
    '''This Task retrieves IDs of diseases and drugs and writes them to a file.
    Index of the DataFrame remains unchanged'''
    
    def output(self):
        
        return luigi.LocalTarget('id_table.hdf')
    
    
    def run(self):
        table = pd.read_excel(INPUT, **INPUT_KW)
        disease_ids = table.apply(func.return_disease_ids, axis=1)
        drug_ids = table.apply(func.return_drug_ids, axis=1)
        combined_ids = pd.DataFrame(index=disease_ids.index)
        combined_ids['disease_ids'] = disease_ids
        combined_ids['drug_ids'] = drug_ids
        
        combined_ids.to_hdf('id_table_real.hdf', 'w', mode='w')
        

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
    with open("table_number.pkl", "rb") as number_file:
        table_number = pkl.load(number_file)

    def requires(self):
        return [RetrieveIDs(), GetNegativeSet()]
    
    def output(self):
        table_ids = ["real"] + list(range(self.table_number))
        return [luigi.LocalTarget('direct_{}.hdf'.format(table_id)) 
                for table_id in table_ids] 
    
    def run(self):
        table_ids = ["real"] + list(range(self.table_number))
        for table_id in tqdm(table_ids):
            if table_id == "real":
                id_table = pd.read_hdf('id_table.hdf', 'w')
            else:
                id_table = pd.read_hdf('id_table_neg_{}.hdf'.format(table_id), 'w')
            direct_relations = id_table.apply(func.find_direct_relations, axis=1) 
            direct_relations.to_hdf('direct_{}.hdf'.format(table_id), 'w', mode='w')
         
class RetrieveDirectRelationInfo(luigi.Task):
    '''This Task retrieves important information on direct relations.'''
    with open("table_number.pkl", "rb") as number_file:
        table_number = pkl.load(number_file)
    
    def requires(self):
        return FindDirect()
    
    def output(self):
        table_ids = ["real"] + list(range(self.table_number))
        return [luigi.LocalTarget('direct_short_{}.hdf'.format(table_id)) 
                for table_id in table_ids] 
    
    def run(self): 
        table_ids = ["real"] + list(range(self.table_number))
        for table_id in tqdm(table_ids):
            direct_relations = pd.read_hdf('direct_{}.hdf'.format(table_id), 'w')
            dir_rel_info = direct_relations.apply(func.pull_direct_relations_info) 
            dir_rel_info.to_hdf('direct_short_{}.hdf'.format(table_id), 'w', mode='w')
    
class FindIndirect(luigi.Task):
    '''This Task finds indirect paths between list of IDS in the rows of positive id_table.'''
    with open("table_number.pkl", "rb") as number_file:
        table_number = pkl.load(number_file)

    def requires(self):
        return [RetrieveIDs(), GetNegativeSet()]
    
    def output(self):
        table_ids = ["real"] + list(range(self.table_number))
        table_length = 403
        return [luigi.LocalTarget("{}_{}.json".format(table_id, row_number)) 
                for row_number in range(table_length)
                for table_id in table_ids]
    
    def run(self):
        table_ids = ["real"] + list(range(self.table_number))
        for table_id in table_ids:
            if table_id == "real":
                id_table = pd.read_hdf('id_table.hdf', 'w')
            else:
                id_table = pd.read_hdf('id_table_neg_{}.hdf'.format(table_id), 'w')
            for row_number, row in id_table.iterrows():
                func.get_indirect(row, table_id, row_number)
            

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
            patch2 = func.plot_dens(cat, nondirect_real, color="red", label="nondirect real", ls="dashed") 
            patch3 = func.plot_dens(cat, direct_random, color="blue", label="direct random", ls="solid") 
            patch4 = func.plot_dens(cat, nondirect_random, color="blue", label="nondirect random", ls="dashed") 

            plt.legend(handles = [patch1, patch2, patch3, patch4])
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
        patch2 = func.plot_dens_diversity(nondirect_real, color="red", label="nondirect real", ls="dashed") 
        patch3 = func.plot_dens_diversity(direct_random, color="blue", label="direct random", ls="solid") 
        patch4 = func.plot_dens_diversity(nondirect_random, color="blue", label="nondirect random", ls="dashed") 

        plt.legend(handles = [patch1, patch2, patch3, patch4])
        plt.title("Diversity")
        plt.savefig("./{}_density_pictures/diversity.png".format(self.protocol), format="png")
        plt.close()
        
if __name__ == '__main__':
        luigi.run()
