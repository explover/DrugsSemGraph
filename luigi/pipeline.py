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
        dsets = ["pos", "neg"]
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
        dsets = ["neg", "pos"]
        for dset in tqdm(dsets):
            id_table = pd.read_hdf(f"{self.data_path}/id_table_{dset}.hdf", 'w')
            dset_table = pd.read_csv(f"{self.data_path}/{dset}.csv")
            #id_table.index = range(len(id_table))
            semgroups_filename = f"{self.data_path}/SemGroups.txt"
            feature_table = func.get_feature_table(id_table, dset, 
                                                   semgroups_filename,
                                                   self.data_path)
            to_append = id_table
            to_append["drug_id"], to_append["disease_id"] = dset_table["drug_id"], dset_table["disease_id"]
            feature_table = pd.merge(to_append, feature_table, right_index=True,
                                     left_index=True)
            feature_table.to_csv(f"{self.data_path}/feature_table_{dset}.csv",
                                 index=False)

if __name__ == '__main__':
        luigi.run()
