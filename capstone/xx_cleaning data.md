### Cleaning Data Tasks Applied
* Regex was used to extract the borough name from a larger string. It is then available to use as a categorical feature
* Regex was used to extract the property size from a larger string. It is then available to use as a numeric feature
* Extensive coding was applied to determine if a property was a 'shared ownership' property, by interrogating several other text fields. This information was used to exclude shared ownership properties (which could not accurately predicted during initial modelling).