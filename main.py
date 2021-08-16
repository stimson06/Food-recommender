from recommendation_system import popularity_recommender
from recommendation_system import item_based_recommender
from Recogniser import FaceIdentify
import pandas as pd

print('[INFO] Loading...')
menu = pd.read_csv('./data/menu.csv')
ratings = pd.read_csv('./data/user_ratings.csv')
order = pd.read_csv('./data/order_check.csv')

# Identifying the customer through face recognition
face = FaceIdentify()
person = face.detect_face()
person = max(set(person), key = person.count) # most occured name
user = person # user

# Popularity recommendations
#recommender = popularity_recommender()
#popular_items  = recommender.is_popular(menu, ratings, top = 4)
#print('Popular Here :\n',popular_items)

# Personalised recommendations (class)
item_recommender = item_based_recommender()
personal = item_recommender.personalized_recommendation(user, menu, order, ratings)
print("\nRecommendation for "+user+' are\n',personal)
