import luigi

import requests, json
import pandas as pd
import func #the module containing all functions


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
        
        
class FindDirect(luigi.Task):
    '''This Task finds direct relations between lists of IDs in the rows of id_table.'''
    
    def requires(self):
        return RetrieveIDs()
    
    def output(self):
        return luigi.LocalTarget('direct.hdf')
    
    def run(self):
        id_table = pd.read_hdf('id_table.hdf', 'w')
        direct_relations = id_table.apply(func.find_direct_relations, axis=1)
        
        direct_relations.to_hdf('direct.hdf', 'w', mode='w')

            
class RetrieveDirectRelationInfo(luigi.Task):
    '''This Task retrieves important information on direct relations.'''
    
    def requires(self):
        return FindDirect()
    
    def output(self):
        return luigi.LocalTarget('direct_short.hdf')
    
    def run(self):
        direct_relations = pd.read_hdf('direct.hdf', 'w')
        
        dir_rel_info = direct_relations.apply(func.pull_direct_relations_info)
        dir_rel_info.to_hdf('direct_short.hdf', 'w', mode='w')
    
    
class FindIndirect(luigi.Task):
    '''This Task finds indirect paths between list of IDS in the rows of positive id_table.'''
    
    def requires(self):
        return [RetrieveIDs(), GetNegativeSet()]
    
    def output(self):
        table_number = 10
        table_ids = ["real"] + list(range(table_number))
        table_length = 403
        return [luigi.LocalTarget("{}_{}.json".format(table_id, row_number)) 
                for row_number in range(table_length)
                for table_id in table_ids]
    
    def run(self):
        table_number = 10
        table_ids = ["real"] + list(range(table_number))
        for table_id in table_ids:
            if table_id == "real":
                id_table = pd.read_hdf('id_table.hdf', 'w')
            else:
                id_table = pd.read_hdf('id_table_neg_{}.hdf'.format(table_id), 'w')
            for row_number, row in id_table.iterrows():
                func.get_indirect(row, table_id, row_number)
            

class GetNegativeSet(luigi.Task):
    '''This Task produces certain number of drug-disease negative sets.'''
    
    def requires(self):
        return RetrieveIDs()
    
    def output(self):
        table_number = 10
        return [luigi.LocalTarget("id_table_neg_{}.hdf".format(table_id)) 
                                  for table_id in range(table_number)]
        
    def run(self):
        table_number = 10
        id_table = pd.read_hdf('id_table.hdf', 'w')
        func.randomize_drugs_diseases(table_number, id_table)

class PlotSemanticCategory(luigi.Task):
    '''This Task plots distributions of normalized counts within
    semantic categories.'''
    protocol = "semanticCategory"
    #def requires(self):
        #return FindIndirect()

    #def output(self):
     #   _, __, cats = self.get_all()
      #  return [luigi.LocalTarget("./new_pictures/{}.png".format(cat)) 
       #         for cat in cats]

    def run(self):
        random_table_counters, nonrandom_counter, cats = self.get_all()
        for cat in cats:
            func.plot_both(cat, nonrandom_counter, random_table_counters)

    def get_all(self):
        id_table = pd.read_hdf('id_table.hdf', 'w') 
        random_table_counters = func.get_counters(10, protocol=self.protocol)
        nonrandom_counter = func.get_sem_cat_counter_list(id_table, "real", protocol=self.protocol)
        random_counter = list()
        for counter in random_table_counters:
            random_counter = random_counter + counter
        cats = set()
        for counter in random_counter + nonrandom_counter:
                cats = cats | set(counter.keys())
        return random_table_counters, nonrandom_counter, cats

class PlotSemanticType(PlotSemanticCategory):
    '''This Task plots distributions of normalized counts within
    semantic types.'''
    protocol = "semanticType"
    
        
class PlotDiversity(luigi.Task):
    '''This Task plots distributions of normalized diversity.'''

    def requires(self):
        return FindIndirect()

    def output(self):
        return luigi.LocalTarget("./pictures/diversity.png")

    def run(self): 
        id_table = pd.read_hdf('id_table.hdf', 'w') 
        random_table_counters = func.get_counters(20, diversity=True)
        nonrandom_counter = func.get_sem_cat_counter_list(id_table, "real", diversity=True)
        func.plot_diversity(nonrandom_counter, random_table_counters)
        

if __name__ == '__main__':
        luigi.run()
