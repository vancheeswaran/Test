import spotipy
from spotipy.oauth2 import SpotifyClientCredentials 
client_id = "0054a24f2fc643c69d56d020dd5f70be"
client_secret = "98b4a4b772ad4eca934a92ca60c246a0"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 

name = ["Micheal Jackson","pitbull","Christina","Elvis Presley"]
result = sp.search(name) 
result['tracks']['items'][1]['artists']