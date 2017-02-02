#coding=utf8
import numpy as np
import time
import pickle
from itertools import compress, permutations


class Economy(object):
    """
    Matsumaya, Kiyotaki & Matsui's model of 
    international currency with indivisible output
    (1993).
    """ 
    
    def __init__(self, parameters):
        
        assert parameters["nb"] % parameters["nb_country"] == 0
        assert 0 <= parameters["c"] < parameters["u"]
        
        self.nb = parameters["nb"]
        self.nb_type = parameters["nb_type"]
        self.nb_country = parameters["nb_country"] #Total of countries
        self.c = parameters["c"] #Production utility (meaning production's cost)
        self.u = parameters["u"] #Consumption utility 
        self.growth = parameters["growth"] #Growth rate
        self.r = parameters["r"] #Time preference
        self.money = parameters["money"] #fraction of money gave to newborn agent
    
        self.switch_type = {0: 1, 1: 2, 2: 0}
        self.switch_country = {1: 2, 2: 1}
       
        
        self.type = np.zeros(self.nb)
                                       
        self.currency = np.zeros(self.nb)
        self.nationality = np.zeros(self.nb)

        self.value =  np.array([[   0 ,  0   , 0    ],
                               
                               [parameters["v"]["1_0"],
                                parameters["v"]["1_1"],
                                parameters["v"]["1_2"]],
                               
                               [parameters["v"]["2_0"], 
                                parameters["v"]["2_1"],
                                parameters["v"]["2_2"]]]) #Advantage of being a seller, buyer
                                                                #depending on the currency
        self.alpha = np.array([[   0 ,  0   , 0    ],
                               [0, 
                                parameters["alpha"]["1_1"],
                                parameters["alpha"]["1_2"]],
                               [0, 
                                parameters["alpha"]["2_1"],
                                parameters["alpha"]["2_2"]]])
        
        self.set_up()

    #-----------------------------------------------------------------------------------------#
    @property
    def equilibrium(self):
        """returns current equilibrium state""" 
        
        equi_list = [[]]
        cond = self.nationality != self.currency  #Check if one agent i holds money j =/= i
        cond2 = self.currency != 0               #This currency has to be different from 0
        match = cond*cond2
        
        for i in range(self.nb_country):
            if (i + 1) in self.nationality[match]:
                equi_list.append([i + 1, True])
            else:
                equi_list.append([i + 1, False])
        
        assert len(equi_list) == self.nb_country + 1
        
        return equi_list 

    #-----------------------------------------------------------------------------------------#
    def set_up(self):
        """fills nationality array and type array""" 
        
        self.nationality = np.array(np.split(self.nationality, self.nb_country))
        
        self.type = np.array_split(self.nationality[0], self.nb_type)       #split arrays 
        self.type = np.array([self.type for i in range(self.nb_country)])        
        
        for i in range(self.nb_country):                                    #fill arrays with
            self.nationality[i][:]= i + 1
            for j in range(self.nb_type):
                self.type[i][j][:] = j
        
        for i in range(self.nb_country):
            self.type = np.concatenate([i for i in self.type])              #concatenate both of them 
                                                                            #in order to get two    
        self.nationality = np.concatenate([i for i in self.nationality])    #singles array without
                                                                            #subdivisions

    #-----------------------------------------------------------------------------------------#
    def increase_population(self):
        """returns N = nb of country arrays of newborn agents"""
    
        nb_newborn = int(self.growth * self.nb)
        nb_newborn_per_country =  int(nb_newborn / self.nb_country)
        self.nb += nb_newborn_per_country*self.nb_country

        newborn = np.array([np.zeros(nb_newborn_per_country)
                   for i in range(self.nb_country)])
        
        return newborn 
        
    #-----------------------------------------------------------------------------------------#
    def inject_money(self, newborn):
        """each time in economy, governments give money to 
        a fraction of newborn citizens, this method returns newborn array
        which has a part of currency holders in it"""
        
        fraction = (int(len(newborn[0]) * self.money))

        for i in range(len(newborn)):
            newborn[i][0:fraction] = i + 1
        
        return newborn 

    #-----------------------------------------------------------------------------------------#
    def add_types(self, newborn):
        """defines types of newborn citizens"""
        
        types = np.array_split(newborn[0], self.nb_type)
        types = np.array([types for i in range(self.nb_country)])
        
        for country in range(len(types)):
            for idx in range(self.nb_type):
                types[country][idx][:] = idx
                self.type = np.append(self.type, types[country][idx])
        
        assert self.nb == len(self.currency) == len(self.nationality) == len(self.type)
        assert len(np.where(self.type == 0)) ==  len(np.where(self.type == 1)) \
               == len(np.where(self.type == 2)) 

    #-----------------------------------------------------------------------------------------#
    def add_newborn(self, newborn):
        """add newborn citizen to population by merging arrays"""
        
        for i in range(len(newborn)):
            currency = np.array(newborn[i])
            np.random.shuffle(currency)
            self.currency = np.append(self.currency, currency)
            newborn[i].fill(i + 1)
            self.nationality = np.append(self.nationality, newborn[i])
        
        assert len(np.where(self.currency == 0)) ==  len(np.where(currency == 1)) 
        assert len(np.where(self.nationality == 1)) == len(np.where(self.nationality == 2)) 
    #-----------------------------------------------------------------------------------------#
    def get_sellers_and_buyers(self):
        """get idx of: fraction of buyers from i with currency 
           j =/= i and currency j == j (mii and mij). Also
           get fraction of sellers from all countries (mi0)"""
        
        buyers = np.zeros(self.nb_country, dtype=np.ndarray)
        sellers = np.zeros(self.nb_country, dtype=np.ndarray)
        
        sum_buyers = 0
        sum_sellers = 0
        
        for i in range(self.nb_country):

            buyers[i] = (self.currency == i + 1)         
            sellers[i] = (self.currency == 0 )*(self.nationality == i + 1)
            
            buyers[i] = np.array(list(compress(range(len(buyers[i])), buyers[i])))
            
            sellers[i] = np.array(list(compress(range(len(sellers[i])), sellers[i])))
            
            sum_buyers += len(buyers[i])  
            sum_sellers += len(sellers[i])
        
        assert (sum_buyers +  sum_sellers) == self.nb
        return (buyers, sellers)

    #-----------------------------------------------------------------------------------------#
    def agents_random_selection(self, buyers, sellers):
        """retrieves randomly agents indices matching conditions ii and ij"""

        buyers_match_ii =  np.zeros(self.nb_country, dtype=np.ndarray)
        buyers_match_ij =  np.zeros(self.nb_country, dtype=np.ndarray)   #we pick with alpha rate
        sellers_match_ii = np.zeros(self.nb_country, dtype=np.ndarray)   #which seller will encounter 
        sellers_match_ij =  np.zeros(self.nb_country, dtype=np.ndarray)
                                                                        #an agent holding foreign money, 
                                                                    #as well as local money.
        for i, j in permutations([i for i in range(self.nb_country)], 2):
            buyers_match_ij[i] = \
                        np.random.choice(buyers[i], buyers[i].size*self.alpha[i+1][j+1], replace=False)
            sellers_match_ij[i] = \
                        np.random.choice(sellers[j], buyers_match_ij[i].size, replace=False)
            
             
            pop_buyers = np.in1d(buyers[i], buyers_match_ij[i], invert=True)
            pop_sellers = np.in1d(sellers[j], sellers_match_ij[i], invert=True)

            buyers[i] = buyers[i][pop_buyers]
            sellers[j] = sellers[j][pop_sellers]
        
        for i in range(self.nb_country):
            buyers_match_ii[i] = \
                    np.random.choice(buyers[i], buyers[i].size*self.alpha[i+1][i+1], replace=False)
            sellers_match_ii[i] = \
                    np.random.choice(sellers[i], buyers_match_ii[i].size, replace=False)
            
            pop_buyers = np.in1d(buyers[i], buyers_match_ii[i], invert=True)
            pop_sellers = np.in1d(sellers[i], sellers_match_ii[i], invert=True)

            buyers[i] = buyers[i][pop_buyers]
            sellers[i] = sellers[i][pop_sellers]

        
        return { "b_m_ii": buyers_match_ii,
                 "b_m_ij": buyers_match_ij, 
                 "s_m_ii": sellers_match_ii, 
                 "s_m_ij": sellers_match_ij }

    #-----------------------------------------------------------------------------------------#
    def agents_random_matching(self, result): 
        """match agents"""
        
        for i in range(len(result["b_m_ij"])):
            for idx in range(len(result["s_m_ij"][0])):
                self.make_choice_and_exchange(result["b_m_ij"][i][idx], result["s_m_ij"][i][idx])
                 
        for i in range(len(result["b_m_ii"])):
            for idx in range(len(result["s_m_ii"][0])):
                self.make_choice_and_exchange(result["b_m_ii"][i][idx], result["s_m_ii"][i][idx])
   
   #-----------------------------------------------------------------------------------------#
    def make_choice_and_exchange(self, id_buyer, id_seller):

        buyer_acceptance = self.type[id_buyer] == self.switch_type[self.type[id_seller]]
        seller_acceptance = self.currency[id_buyer] == self.nationality[id_seller]
        
        if not seller_acceptance:
            seller_acceptance = (self.value[self.nationality[id_seller], self.currency[id_buyer]] 
                        - self.c) > self.value[self.nationality[id_seller], 0]
        
        if buyer_acceptance and seller_acceptance:
            self.currency[id_buyer], self.currency[id_seller] = \
                    self.currency[id_seller], self.currency[id_buyer]

  #-----------------------------------------------------------------------------------------#
    def update_values(self):
        
        for country in [1, 2]:
            for currency in [0, 1, 2]:
                
                cond_mii  = (self.nationality == self.currency)\
                          *(self.currency != 0)*(self.nationality == country)

                cond_mij  = (self.currency != self.nationality)*(self.currency != 0)\
                            *(self.nationality == country)
                
                cond_mi0  = (self.currency != self.nationality)*(self.currency == 0)\
                            *(self.nationality == country)
                
                cond_mji = (self.currency != self.nationality)*(self.currency != 0)\
                            *(self.nationality == self.switch_country[country])
                
                cond_mjj = (self.nationality == self.currency)*(self.currency != 0)\
                           *(self.nationality == self.switch_country[country])
                
                cond_mj0 =  (self.currency != self.nationality)*(self.currency == 0)\
                            *(self.nationality == self.switch_country[country])


                mii = len(self.currency[cond_mii])/self.nb
                mij = len(self.currency[cond_mij])/self.nb
                mi0 = len(self.currency[cond_mi0])/self.nb
                mji = len(self.currency[cond_mji])/self.nb
                mjj = len(self.currency[cond_mjj])/self.nb
                mj0 = len(self.currency[cond_mj0])/self.nb
                

                if currency == 0:
                    self.value[country, currency] = self.r\
                                                    *( self.alpha[country, country]
                                                    * mii 
                                                    + self.alpha[country, self.switch_country[country]]
                                                    * mji)\
                                                    *(self.value[country, country]
                                                    - self.value[country, currency]
                                                    - self.c)\
                                                    +(self.alpha[country, country]
                                                    *mij
                                                    +self.alpha[country, self.switch_country[country]]
                                                    *mjj)\
                                                    *self.equilibrium[country][1]\
                                                    *(self.value[country, self.switch_country[country]]
                                                    - self.value[country, currency]
                                                    - self.c)
                elif currency == 1:
                    self.value[country, currency] = self.r\
                                                    *(self.alpha[country, country]
                                                    *mi0
                                                    +self.alpha[country, self.switch_country[country]]
                                                    *mj0
                                                    *self.equilibrium[self.switch_country[country]][1])\
                                                    *(self.u
                                                    + self.value[country, 0]
                                                    - self.value[country, country])
                    
                else:                                    
                    self.value[country, currency] = self.r\
                                                    *(self.alpha[country, country]
                                                    *mi0
                                                    *self.equilibrium[country][1]
                                                    +self.alpha[country, self.switch_country[country]]
                                                    *mj0)\
                                                    *(self.u
                                                    + self.value[country, 0]
                                                    - self.value[country, self.switch_country[country]])

                
    
    @staticmethod
    def main():
    
        parameters = { "c": 0.1,
                   "u": 0.2,
                   "r": 0.1,
                   "money": 0.3,
                   "alpha": {"1_1": 0.20,
                             "1_2": 0.8,
                             "2_1": 0.8,
                             "2_2": 0.20
                             },
                   "v": {"1_0":  0.5,
                          "1_1": 0.5,
                          "1_2": 0.5,
                          "2_0": 0.5,
                          "2_1": 0.5,
                          "2_2": 0.5
                             },
                   "nb_type": 3,
                   "nb_country": 2,
                   "nb": 600,
                   "growth": 0.1
                }
    
        Eco = Economy(parameters)
    
        while True: 
            newborn = Eco.increase_population()
            newborn = Eco.inject_money(newborn)
            Eco.add_newborn(newborn)
            Eco.add_types(newborn)
            result = Eco.get_sellers_and_buyers()
            result = Eco.agents_random_selection(result[0], result[1])
            Eco.agents_random_matching(result)
            Eco.update_values()
            print(Eco.value)
            print(Eco.equilibrium)
            

if __name__ == '__main__':
    
    Economy.main()
