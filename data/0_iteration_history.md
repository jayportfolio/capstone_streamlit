## Dataset iteration history



1. n/a
2. First version.
   1. Features are:
      * location.latitude
      * location.longitude
      * bedrooms
      * bathrooms
      * nearestStation
      * tenure
      * tenureType
   2. Target is Price. 
3. samples which are shared ownership removed. latitude_deviation from center added, longitude_deviation from centre added.
4. removed samples where the distance to the nearest station is an outlier. Additionally, extreme outliers for bedroom/bathroom/longitude removed, subsequent to exploratory data analysis
5. removed location.latitude/location.longitude and retained latitude_deviation/longitude_deviation, because they were highly correlated, and the latter was better correlated to price (and had a linear trend rather than quadratic)
6. 
   1. removed samples where the distance to the nearest station is an outlier. Additionally, extreme outliers for bedroom/bathroom/longitude removed, subsequent to exploratory data analysis
   2. outlier detection is more stringent than for version 4, and uses tukey and kde outlier detection to detect outliers.
7. added listing_date as a feature, to check impact of date-based feature (rejected)
8. not used (but included keyFeatures in preparation for later iterations)
9. added features extracted from property description. 10 features (TBC)
10. added features extracted from property description 50 features (TBC)
11. added features extracted from property description. Additional method of text extraction for feature used (TBC)