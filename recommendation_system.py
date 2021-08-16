from enum import unique
import pandas as pd 
import numpy as np
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class popularity_recommender():
    """ Helps to find the popular dish in the hotel with the help of menu and 
        ratings given by the user 
        
        Condition used:
        The dish must have vote count more the 80% of the total votes.
    """
    def __init__(self):
        self.top = 0 
    
    def is_popular(self,menu, ratings, top = 5):
        """ For computing the popular dishes in the hotel

        Args:
            menu (.csv): The dishes present in the hotel.
            ratings (.csv): The ratings given by the user.
            top (int, optional): Helps in getting the top most items. Defaults to 5.

        Returns:
            top5(list): list of popular items.
        """
        
        # Creating a temporary dataframe out of rating
        ratings_detail = pd.DataFrame()
        ratings_detail['item_id']=[]
        ratings_detail['vote_counts']=[]
        ratings_detail['avg_rating']=[] 
        for id in  ratings.item_id.sort_values().unique():
            temp_df = ratings[ratings.item_id == id]
            ratings_detail.loc[id] = [int(id), int(len(temp_df)), round(temp_df.ratings.mean(),2)]
        m = ratings_detail['vote_counts'].quantile(0.80) # 
        data = self.merge_data(menu, ratings_detail)
        popular_items = data.copy().loc[data['vote_counts'] >= m]
        popular_items = popular_items.sort_values('avg_rating', ascending=False)
        top5 = popular_items['Name'].head(top)
        return list(top5)

    # Merging the menu and ratings_details
    def merge_data(self, menu, ratings_detail):
        return pd.merge(menu, ratings_detail, on = 'item_id' )

class item_based_recommender():
    """ Personalized recommendation based on each user previous orders.
        The recommendations will be the dishes similar to the previous orders (dishes the customer hasn't tried before)
    """
    
    # Name of the food with the help of index value
    def get_name_from_index(self,menu, index):
        return menu[menu.index == index].Name.values[0]

    # Index value of the food with help of food name
    def get_index_from_name(self,menu, Name):
        return menu[menu.Name == Name].index.values[0]
    
    # Combining the all the features into one column
    def combine_features(self, row):
        return row['Diet']+" "+row['Course']+" "+row['Ingredients']+" "+row['Flavour']+" "+row['Cusine']

    # Replacing the the commas in the combined feature column
    def replace(self, row):
        return row['combine_features'].replace(',','')
    
    def is_similar(self, menu, likes, top = 5):
        """ Finds the similar foods in accordance with the food ordered by the customer

        Args:
            menu (.csv): The dishes present in the hotel.
            likes (string): Previously ordered food by the customer.
            top (int, optional): Top 5 recommendation. Defaults to 5.

        Returns:
            similar_foods(list): Dishes that are similar to the food of customer.
            [Spelling_error]: Retruns the list with similar words
        """
        if likes in menu.Name.unique():
            
            menu['combine_features']= menu.apply(self.combine_features,axis=1) # Combining the feature columns
            menu['combine_features'] = menu.apply(self.replace, axis=1) # Replacing the comma in b/w
            cv = CountVectorizer() # CountVectorizer() object
            count_matrix = cv.fit_transform(menu['combine_features'])
            cosine_sim = cosine_similarity(count_matrix) 
            customer_likes = likes.lower()
            dish_index = self.get_index_from_name(menu, customer_likes)
            similar_dishes = list(enumerate(cosine_sim[dish_index]))
            sorted_similar_dishes = sorted(similar_dishes,key=lambda x:x[1],reverse=True)[1:] # Sorting in descending order
            i=0
            similar_foods=[]

            for element in sorted_similar_dishes:
                similar_foods.append(self.get_name_from_index(menu, element[0]))
                i=i+1
                if i>top-1:
                    break
            return similar_foods
        else:
            
            return ("Did you mention? {}".format((menu.Name[menu['Name'].str.contains(likes)]).tolist()))
             
    def personalized_recommendation(self, customer, menu, order, rating):
        """ Checks the frequency of the customer and creates the 
            the list of food based on the previous orders, course 
            and cusine. The list is then passed to is_similar funtion
            to personalize the recommendation.
            
            If there is no details of the customer in the order.csv
            recommendation are the popular dishes in the hotel.

        Args:
            customer (string): The name of the customer.
            menu (.csv): Dishes in the hotel.
            order (.csv): order details of the customer.

        Returns:
            recommendation(list): Recommendations for the customer.
        """
        
        menu['Name'] = menu['Name'].str.lower()
        menu['Name'] = menu['Name'].str.strip()
        order['Name'] = order['Name'].str.lower()
        order['Name'] = order['Name'].str.strip()
        
        #Function to add the key value
        def add(dictonary, key, value): 
            dictonary[key]=value 
        
        # Unkown customer
        user = customer
        if user == 'Unknown':
            recommender = popularity_recommender()
            recommendation = recommender.is_popular(menu, rating)
            
            return recommendation
            
        else:
            #Confining the dataframe
            df = order.loc[order['Name'] == user]
            
            #Flattening the foods_ordered
            foods_of_customer = []
            for food in df['Foods ordered']:
                foods = food.split(',')
                foods_of_customer.append(foods)
            foods_of_customer = list(np.concatenate(foods_of_customer).flat) # 2d array to 1d array
            
            #String manipulations
            for i in range(len(foods_of_customer)):
                foods_of_customer[i] = foods_of_customer[i].lower().rstrip().lstrip()
            food_cpy = foods_of_customer.copy()
            
            # Customer who is frequent and have common foods ordered
            if (len(df)>2):
                count = Counter(foods_of_customer)
                foods_of_customer = list([item for item in count if count[item]>1]) # list of food for frequent customer and having common in some of his vist
                
                # Customer who is frequent and have no foods in common
                if (len(foods_of_customer)==0):
                    course_dict = {}
                    cusine_dict = {}
                    for i in food_cpy:
                        details = menu.loc[menu['Name'] == i]
                        course_val = details.iloc[0]['Course'] # Course (starter, main course, soup, dessert)
                        add(course_dict, i, course_val)
                        cusine_val = details.iloc[0]['Cusine'] # Cusine (North indian, South indian, Chinese)
                        add(cusine_dict, i,cusine_val)
                        
                        dict_1 = {}
                        for key, value in course_dict.items():
                            dict_1.setdefault(value, set()).add(key)
                            
                        result_1 = set(chain.from_iterable(
                                values for key, values in dict_1.items()
                                if len(values) > 1))
                        course = list(result_1)
                        
                        dict_2 = {}
                        for key, value in cusine_dict.items():
                            dict_2.setdefault(value, set()).add(key)
                            
                        result_2 = set(chain.from_iterable(
                                values for key, values in   dict_2.items()
                                if len(values) > 1))
                        cusine = list(result_2)
                        
                        intersection = set(cusine).intersection(course)
                        foods_of_customer = list(intersection)
                
                # Rare customer (visited once)
                else:
                    foods_of_customer = list(set(foods_of_customer))
            
            # Getting the similar foods
            if foods_of_customer is not None:
                recommendations = []
                for i in range(len(foods_of_customer)):
                    fav_food = foods_of_customer[i]
                    similar_foods = self.is_similar(menu, fav_food)
                    recommendations.append(similar_foods[:2])
            recommendations = list(np.concatenate(recommendations).flat)
            return recommendations