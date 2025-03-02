# Coachella Artist List Cluster Analysis
The purpose of this project is to analyze the Coachella lineup and use data to cluster artists together.

As of Feb 9, 2025 the artist list is based on the 2025 Coachella lineup. Hopefully you can use this tool to discover new artists based on your personal music tastes!

### Data Scraping
I used Selenium to scrape artist tile data for each artist in the Coachella lineup page. I included all artists 
regardless of day. The scraper pulled 149 artist results, which matches the number on the lineups page.    

There are some acts that are a combination of artists, these artists have been separated in th list.  For example:
- Dixon and Jimi Jules
- Gustavo Dudamel & La Phil
- Mind Against x Massano
- Pete Tong x Amed Spins
- Seun Kuti & Egypt 80

So actually 154 unique artists. 

### Making a Spotify Connection
To get artist and track data from Spotify, I made a Spotify Authentication class to make a connection. This requires 
getting an access / refresh token from the Spotify API to query for artist and track data. To get an access token, you 
will need to create an app on the Spotify Developers web site; run SpotifyAuth(), and you will be prompted to save your
client ID, client secret, and redirect URL (all from Developer site). 

#### Extracting Artist IDs
We need to get the artist IDs to get the top tracks of every artist. To fetch the artist ID, I ran a search using the 
scraped artist names. Most names returned the correct artist and id in Spotify. Some overrides had to have been made to 
map the correct artist.
- GEL (returned GELO)

#### Extracting Top Songs
For simplicity, I have limited the scope of songs for each artist to just be top tracks. Top tracks can be pulled by
using the artist id, and it will return up to the top 10 ten tracks of a given artist. Some notes:
- Features - an artist's top track could be a feature on someone else's song (e.x Open Arms (feat. Travis Scott)) 
- Unique songs - songs may only appear once for an artist. Duplicates are excluded. (i.e Peach is a top track for both Sammy Virji and Salute, dropped from Sammy top tracks)
  - There were 14 duplicate songs
- There can only be one song for an artist (no multiple remixes)
  - There were 79 other songs (repeated song with different track id, remixes)

Most artists have ten top tracks. There are a few that have nine songs, and some less due to the exclusion rules above.
In total, our scope of songs is 1,444.

#### Getting Song URLs from Youtube
Using the Spotify API, we are able to extract track name, popularity, artist, etc. As of starting this project, the audio
features have been deprecated from the API (i.e danceability, loudness, etc.). As an alternative, I have decided to use
Essentia models to extract audio features from the audio files themselves to perform this cluster analysis. To do so, I 
had to get the youtube urls of every song which then could be downloaded. 

- some video duration and views are very small, and vice versa, had to validate
  - The shortest track is by circle jerks (23 seconds). The longest is by Talon, which is Autobahn remastered.
- views may not be as representative as official audio could be much smaller in views than the actual music video.
