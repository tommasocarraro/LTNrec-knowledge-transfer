import pandas as pd

genres = ("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
          "Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
          "Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film",
          "Western Film")

mr_entities = pd.read_csv("./datasets/mindreader/entities.csv")
mr_entities_dict = mr_entities.to_dict("records")
mr_ratings = pd.read_csv("./datasets/mindreader/ratings.csv")

mr_genres = {entity["uri"]: entity["name"] for entity in mr_entities_dict if "Genre" in entity["labels"]}

mr_genre_ratings_original = mr_ratings[mr_ratings["uri"].isin(mr_genres) & mr_ratings["sentiment"] != 0].reset_index().to_dict("records")

triples = pd.read_csv("./datasets/mindreader/triples.csv").to_dict("records")
genre_subclass = [(mr_genres[triple["head_uri"]], mr_genres[triple["tail_uri"]])
                  for triple in triples
                  if triple["head_uri"] in mr_genres and mr_genres[triple["head_uri"]] not in genres
                  if triple["tail_uri"] in mr_genres and mr_genres[triple["tail_uri"]] in genres
                  if triple["relation"] == "SUBCLASS_OF"]

print(genre_subclass)
pd.d()

rating_per_genre = {}
for genre_rating in mr_genre_ratings_original:
    if mr_genres[genre_rating["uri"]] not in rating_per_genre:
        rating_per_genre[mr_genres[genre_rating["uri"]]] = 1
    else:
        rating_per_genre[mr_genres[genre_rating["uri"]]] += 1

sorted_rating_per_genre = sorted(rating_per_genre.items(), key=lambda x: x[1])
sorted_rating_per_genre.reverse()
print(sorted_rating_per_genre)

("Action Film", "Adventure Film", "Animated Film", "Children's Film", "Comedy Film", "Crime Film",
"Documentary Film", "Drama Film", "Fantasy Film", "Film Noir", "Horror Film", "Musical Film",
"Mystery Film", "Romance Film", "Science Fiction Film", "Thriller Film", "War Film",
"Western Film")

# film based on literature
# romantic comedy
# superhero
# comedy-drama
# speculative film
# spy film
# disaster
# buddy
# crime thriller
# dystopian
# teen
# biographical
# post-apocalyptic
# action comedy
# drama
# martial arts
# lgbt
# zombie film -> horror
# parody -> comedy
# historical period
# science fiction action film -> science fiction
# fantasy
# monster
# christmas film
# comedy horror
# documentary
# costume drama
# neo noir
# comedy thriller
# historical film
# vampire
# thriller
# science fiction
# survival
# time travel
# prison
# zombie comedy
# traditionally animated
# sports
# political thriller
# gothic
# girls with guns
# rape and revenge
# psycological
# black comedy
# nature documentary
# action thriller
# family film
# animated film
# gangster
# alien invasion
# comic science fiction
# splatter
# crime-comedy
# psycological horror