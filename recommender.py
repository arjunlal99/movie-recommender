import pandas as pd
import csv
import requests
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

api_key = '4289f8cc07862951961a23e15e408019'


df = pd.DataFrame(columns = ['title','directors','actors','genres','overview'])

filename = '/home/kali/ml-latest-small/movies.csv'

#check whether a movie exists in the database
def if_exists(movie_name):
    url = "https://api.themoviedb.org/3/search/movie?api_key=" + api_key + "&query=" + movie_name +"&page=1"
    r = (requests.get(url)).json() 

    if r["total_results"] == 0:
        return False
    else:
        return True


#get the tmdb id from the movie name
def tmdb_id(movie_name):
    url = "https://api.themoviedb.org/3/search/movie?api_key=" + api_key + "&query=" + movie_name +"&page=1"
    r = (requests.get(url)).json()
    return str(r['results'][0]['id'])

#format the title to remove the year at the end
def format_title(title):
    l = title.split()
    l.pop()
    return ' '.join(l)

#format the genres in the csv file
def format_genre(genres):
    l = genres.split('|')
    return ' '.join(l)

#get title, genres, overview, actors and directors...id is the tmdb id and should be string
def get_details(id):
    details = {
        "genres" : [],
        "overview": "",
        "actors": [],
        "directors": []
    }
    # first get title genres and overview
    url = "https://api.themoviedb.org/3/movie/"+ id + "?api_key=" + api_key
    r = requests.get(url)
    r = r.json()

    #overview
    details["overview"] = r["overview"]
    
    #genre
    for i in r['genres']:
        details['genres'].append(i['name'])
    


    #then get cast and crew
    url = "https://api.themoviedb.org/3/movie/" + id + "/credits?api_key=" + api_key
    r = requests.get(url)
    r = r.json()

    #directors
    for i in r['crew']:
        if i['job'] == 'Director':
            details['directors'].append(i['name'])
    
    #actors
    for i in range(5):
        details['actors'].append(r['cast'][i]['name'])


    return details
#print(type(tmdb_id("the shawshank redemption")))
#print(format_title('Ace Ventura: When Nature Calls (1995)'))
#print(get_details('278'))


with open(filename, 'r') as file:
    reader = csv.reader(file)
    x = 0
    for row in reader:
        

        if x == 10:
            break
        x = x+1

        if if_exists(format_title(row[1])):
            title = format_title(row[1])
            genres = format_genre(row[2])
            details = get_details(tmdb_id(title))
            
            overview = details['overview']

            for genre in details['genres']:
                genres = genres + ' ' + genre 

            directors = ''
            for director in details['directors']:
                directors = directors + ' ' + ''.join(director.split())
            
            actors = ''
            for actor in details['actors']:
                actors = actors + ' ' + ''.join(actor.split())
    #        print(title)
     #       print(directors)
      #      print(actors)
       #     print(genres)
        #    print(overview)
         
            df2 = pd.DataFrame([[title,directors,actors,genres,overview]],columns = ['title','directors','actors','genres','overview'])

            df = df.append(df2)
            print (df2)
            
print("Original Dataframe successfully built")
#dataframe containing bag_of_words
keywords_df = pd.DataFrame(columns = ['title','bag_of_words'])
#bag_of_words...

#extract bag of words from each row
def extract_bag_of_words(row):
    bag_of_words = row[1].lower() + ' ' + row[2].lower() + ' ' + row[3].lower()
    #keyword extraction from overview using Rake
    r = Rake()
    r.extract_keywords_from_text(row[4].lower())

    key_words_dict_scores = r.get_word_degrees()

    overview_keywords = ' '.join(list(key_words_dict_scores.keys()))

    bag_of_words = bag_of_words + ' ' + overview_keywords

    return bag_of_words


for row_index,row in df.iterrows():
    bag_of_words = extract_bag_of_words(row) 
    
    keywords_df2 = pd.DataFrame([[row[0],bag_of_words]],columns = ['title','bag_of_words'])

    keywords_df = keywords_df.append(keywords_df2)
    print(keywords_df2)


print("bag_of_words dataframe built successfully")

#cosine matrix
count = CountVectorizer()

count_matrix = count.fit_transform(keywords_df['bag_of_words'])

cosine_sim = cosine_similarity(count_matrix,count_matrix)

print(cosine_sim)