Exploratory data engineering showed: 
* Data at the extreme latitudes were more likely to be freehold properties
* Data at the extreme longitudes were more likely to be freehold properties
This could potentially be explained by my choice to cap properties at £600,000


Additionally:
* Freehold properties closer to the latitude mean were more likely to be expensive (further away in either direction were less expensive),
* Freehold properties closer to the longitude mean were more likely to be expensive (further away in either direction were less expensive).

It may be possible for certain ML algorithms to account for the parabolic trending of this feature (such as K Neighest neighbour or a 
neural network with an approximate loss algorithm, but I have taken an early decision to feature engineer two new features to assess if
they improve algorithm performance. They are
* Distance from average latitude
* Distance from average longitude.