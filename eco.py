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
        
        assert parameters["nb"] % parameters["nb_countries"] == 0
        assert 0 <= parameters["c"] < parameters["u"]
        
        self.nb = parameters["nb"]
        self.nb_type = parameters["nb_type"]
        self.nb_countries = parameters["nb_countries"] #Total of countries
        self.c = parameters["c"] #Production utility (meaning production's cost)
        self.u = parameters["u"] #Consumption utility 
        self.growth = parameters["growth"] #Growth rate
        self.r = parameters["r"] #Time preference
        self.money = parameters["money"] #fraction of money gave to newborn agent
    
        self.switch_type = {0: 1, 1: 2, 2: 0}
        self.switch_country = {1: 2, 2: 1}
       
        self.steady_state = [1] * (self.nb_countries + 1)

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
        
        cond = self.nationality != self.currency  #Check if one agent i holds money j =/= i
        cond2 = self.currency != 0               #This currency has to be different from 0
        match = cond*cond2
        print(self.steady_state)
        steady = self.steady_state[1] < 0.1 
         
        return [country in self.nationality[match] and steady for country in [0, 1, 2]]

    #-----------------------------------------------------------------------------------------#
    def set_up(self):
        """fills nationality array and type array""" 
        
        self.nationality = np.array(np.split(self.nationality, self.nb_countries))
        
        self.type = np.array_split(self.nationality[0].copy(), self.nb_type)       #split arrays 
        self.type = np.array([self.type for i in range(self.nb_countries)])        
        
        for i in range(self.nb_countries):                                    #fill arrays with
            self.nationality[i][:]= i + 1
            for j in range(self.nb_type):
                self.type[i][j][:] = j
        
        for i in range(self.nb_countries):
            self.type = np.concatenate([i for i in self.type])              #concatenate both of them 
                                                                            #in order to get two    
        self.nationality = np.concatenate([i for i in self.nationality])    #singles array without

    #-----------------------------------------------------------------------------------------#
    def increase_population(self):
        """returns N = nb of country arrays of newborn agents"""
    
        nb_newborn = int(self.growth * self.nb)
        nb_newborn_per_country =  int(nb_newborn / self.nb_countries)
        self.nb += nb_newborn_per_country * self.nb_countries

        newborn = np.array([np.zeros(nb_newborn_per_country)
                   for i in range(self.nb_countries)])
        
        return newborn 
        
    #-----------------------------------------------------------------------------------------#
    def inject_money(self, newborn):
        """each time in economy, governments give one unit money to 
        a fraction of newborn citizens, in exchange of an equal 
        amount of goods.
        This method returns newborn array
        which has a part of currency holders in it.
        """
        
        fraction = int(len(newborn[0]) * self.money)

        for i in range(len(newborn)):
            newborn[i][0:fraction] = i + 1
        
        return newborn 
    
    #-----------------------------------------------------------------------------------------#
    def add_types(self, newborn):
        """defines types of newborn citizens"""
        
        types = np.array_split(newborn[0], self.nb_type)
        types = np.array([types for i in range(self.nb_countries)])
        
        for country in range(len(types)):
            for idx in range(self.nb_type):
                types[country][idx][:] = idx
                self.type = np.append(self.type, types[country][idx])
        
        assert self.nb == len(self.currency) == len(self.nationality) == len(self.type)
        assert len(np.where(self.type == 0)) ==  len(np.where(self.type == 1)) \
               == len(np.where(self.type == 2)) 

    #-----------------------------------------------------------------------------------------#
    def add_newborn(self, newborn):
        """add newborn citizen to population"""
        
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
        """get the two separates groups in order 
        to make them encounter later"""
        
        nationality_i = self.nationality == 1
        nationality_j = self.nationality == 2

        #get idx for true statements
        nationality_i_idx = list(compress(range(len(nationality_i)), nationality_i))
        nationality_j_idx = list(compress(range(len(nationality_j)), nationality_j))
        
        assert len(nationality_i_idx) == len(nationality_j_idx)
        
        return [nationality_i_idx, nationality_j_idx]
    
    #-----------------------------------------------------------------------------------------#
    def poisson_distribution(self):
        """returns number of meeting for ij, ii randomly 
        picked in a poisson distribution 
        based on sample and average arrival rate"""
        
        nb_of_meeting_ii = []
        nb_of_meeting_ij = []
        
        for i in [1, 2]:
            array = np.random.poisson(self.alpha[i, i] * self.nb_type, self.nb)
            nb_of_meeting_ii.append(np.random.choice(array))
        

        for i in [(1, 2), (2, 1)]:
            array = np.random.poisson(self.alpha[i[0], i[1]] * self.nb_type, self.nb)
            nb_of_meeting_ij.append(np.random.choice(array))

        return {"ii": nb_of_meeting_ii, "ij": nb_of_meeting_ij}

    #-----------------------------------------------------------------------------------------#
    def main_agents_random_matching(self, nationality, meeting_dict):
        
        #ii matching 
        for i, number_of_meeting in enumerate(meeting_dict["ii"]):
            nationality[i] = self.agents_random_matching(nationality[i], 
                                                         nationality[i],
                                                         number_of_meeting)[0]
        #ij matching
        for number_of_meeting in meeting_dict["ij"]:
            nationality[0], nationality[1] = self.agents_random_matching(nationality[0], 
                                                                         nationality[1],
                                                                         number_of_meeting)
    #-----------------------------------------------------------------------------------------#
    def agents_random_matching(self, nationality_1, nationality_2, number_of_meeting):
        """match agent using array of country 1 and country 2 and number of meeting.
        Returns nationality list without picked index"""
        
        #for number of meeting 
        for i in range(number_of_meeting):
            
            #we picks randoms idx and then remove them
            idx_1 = np.random.randint(len(nationality_1))
            agent_idx_1 = nationality_1[idx_1]
            nationality_1.pop(idx_1)
            
            idx_2 = np.random.randint(len(nationality_2))
            agent_idx_2 = nationality_2[idx_2]
            nationality_2.pop(idx_2)
            
            self.make_choice_and_exchange(agent_idx_1, agent_idx_2)
        
        return (nationality_1, nationality_2)
     
   #-----------------------------------------------------------------------------------------#
    def make_choice_and_exchange(self, id_buyer, id_seller):
        """exchange or not"""

        seller_nationality = int(self.nationality[id_seller])
        buyer_nationality =  int(self.nationality[id_buyer])
        
        seller_type = int(self.type[id_seller])
        buyer_type = int(self.type[id_buyer])
        
        buyer_currency = int(self.currency[id_buyer])
        seller_currency = int(self.currency[id_seller])
        
        buyer_acceptance = buyer_type == self.switch_type[seller_type]
        seller_acceptance = buyer_currency == seller_nationality
        
        if not seller_acceptance:
            seller_acceptance = (self.value[seller_nationality, buyer_currency]
                                - self.c) > self.value[seller_nationality, 0]
        
        if buyer_acceptance and seller_acceptance:
            self.currency[id_buyer], self.currency[id_seller] = \
            seller_currency, buyer_currency

  #-----------------------------------------------------------------------------------------#
    def update_values(self):
        
        for country in [1, 2]:
            for currency in [0, 1, 2]:
                
                cond_mii  = (self.nationality == self.currency)\
                            * (self.currency != 0)*(self.nationality == country)

                cond_mij  = (self.currency != self.nationality)*(self.currency != 0)\
                            * (self.nationality == country)
                
                cond_mi0  = (self.currency != self.nationality)*(self.currency == 0)\
                            * (self.nationality == country)
                
                cond_mji = (self.currency != self.nationality)*(self.currency != 0)\
                            * (self.nationality == self.switch_country[country])
                
                cond_mjj = (self.nationality == self.currency)*(self.currency != 0)\
                           * (self.nationality == self.switch_country[country])
                
                cond_mj0 =  (self.currency != self.nationality)*(self.currency == 0)\
                            * (self.nationality == self.switch_country[country])

                mii = len(self.currency[cond_mii]) / self.nb
                mij = len(self.currency[cond_mij]) / self.nb
                mi0 = len(self.currency[cond_mi0]) / self.nb
                mji = len(self.currency[cond_mji]) / self.nb
                mjj = len(self.currency[cond_mjj]) / self.nb
                mj0 = len(self.currency[cond_mj0]) / self.nb
                
                if currency == 0:
                    self.value[country, currency] = self.r\
                                                    * (self.alpha[country, country]
                                                    * mii 
                                                    + self.alpha[country, self.switch_country[country]]
                                                    * mji)\
                                                    * (self.value[country, country]
                                                    - self.value[country, currency]
                                                    - self.c)\
                                                    + (self.alpha[country, country]
                                                    * mij
                                                    + self.alpha[country, self.switch_country[country]]
                                                    * mjj)\
                                                    * self.equilibrium[country]\
                                                    * (self.value[country, self.switch_country[country]]
                                                    - self.value[country, currency]
                                                    - self.c)
                elif currency == country:
                    self.value[country, currency] = self.r\
                                                    * (self.alpha[country, country]
                                                    * mi0
                                                    + self.alpha[country, self.switch_country[country]]
                                                    * mj0
                                                    * self.equilibrium[self.switch_country[country]])\
                                                    * (self.u
                                                    + self.value[country, 0]
                                                    - self.value[country, country])
                    
                else:                                    
                    self.value[country, currency] = self.r\
                                                    * (self.alpha[country, country]
                                                    * mi0
                                                    * self.equilibrium[country]
                                                    + self.alpha[country, self.switch_country[country]]
                                                    * mj0)\
                                                    * (self.u
                                                    + self.value[country, 0]
                                                    - self.value[country, self.switch_country[country]])
                
                #check if steady state equations are statisfied 
                self.steady_state[country] = (self.alpha[country, self.switch_country[country]]
                                             * mi0
                                             * mji)\
                                             - (self.alpha[country, self.switch_country[country]]
                                             * mii
                                             * mj0
                                             * self.equilibrium[self.switch_country[country]])\
                                             + self.growth\
                                             * (self.money
                                             - mii)
                                        
  #-----------------------------------------------------------------------------------------#
    
    @staticmethod
    def main():
    
        parameters = { "c": 0.1,
                   "u": 0.2,
                   "r": 0.2,
                   "money": 0.5,
                   "alpha": {"1_1": 5,
                             "1_2": 10,
                             "2_1": 10,
                             "2_2": 5
                             },
                   "v": {"1_0":  1000,
                          "1_1": 0.5,
                          "1_2": 0.5,
                          "2_0": 1000,
                          "2_1": 0.5,
                          "2_2": 0.5
                             },
                   "nb_type": 3,
                   "nb_countries": 2,
                   "nb": 400,
                   "growth": 0.2
                }
    
        Eco = Economy(parameters)
        i = 0
        while True: 
            newborn = Eco.increase_population()
            newborn = Eco.inject_money(newborn)
            Eco.add_newborn(newborn)
            Eco.add_types(newborn)
            population = Eco.get_sellers_and_buyers()
            nb_of_meeting = Eco.poisson_distribution()
            Eco.main_agents_random_matching(population, nb_of_meeting)
            Eco.update_values()
            print(Eco.value)
            print(Eco.equilibrium)
            i += 1
            if i > 15:
                import pdb; pdb.set_trace()
            
if __name__ == '__main__':
    
    Economy.main()
