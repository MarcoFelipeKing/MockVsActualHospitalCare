# Mock vs Actual Hospital Care
Data behind the journal article: "Why is mock care not a good proxy for predicting hand contamination during patient care?" (https://doi.org/10.1016/j.jhin.2020.11.016)

This dataset is citeable at:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4314760.svg)](https://doi.org/10.5281/zenodo.4314760)

These data are sequential surface contacts during mock patient care and actual hospital care for a variety of types of care: Taking bloods, checking on a patient, taking observations/vitals, IV drip manipulation, fiting a nebuliser, discussing medication by a pharmacist and Drs/MD rounds. This dataset includes observations in single patient rooms (SB) as well as observations in 4-patient multi-bed rooms (MB), whilst  https://doi.org/10.1016/j.jhin.2020.11.016 only uses the SB data.

# Headers in the stacked long format .csv file
``` r
 names(df)<-c("ActivityID", "CareType", "HCWType", "RoomType", "Surface", "Date",	"Time",	"Dev.Date.Time",	"type")
```

## Factor levels for each variable
``` r

$CareType
[1] "Bloods"    "Check"     "IV"        "Nebuliser" "Obs"       "Pharmacy"  "Rounds"   

$HCWType
[1] "AUX"          "DR"           "P"            "PHLEBOTOMIST" "PHYS"         "RN"           "SN"           "STUDN"       

$RoomType
[1] "MB" "SB"

$Surface
 [1] "Alc"            "AlcInside"      "AlcOutside"     "ApronOff"       "ApronOn"        "Bed"            "Bedding"        "BloodObsEq"     "Chair"          "Curtain"        "Door"           "EqMisc"         "EqTray"         "Equipment"     
[15] "GlovesOff"      "GlovesOn"       "GownOff"        "GownOn"         "In"             "IV"             "IVDrip"         "IVStand"        "MedsTrolley"    "Nebuliser"      "Notes"          "ObsTrolley"     "Other"          "OtherFar"      
[29] "OtherNear"      "Out"            "PaperTowel"     "Patient"        "Sharps"         "Sink"           "Soap"           "Stethoscope"    "Syringe"        "Table"          "TabletComputer" "Tap"            "Tray"           "Waste"         
[43] "WindowBlind"    "Wipes"         

$type
[1] "Actual" "Mock"  
```
<sup>Created on 2020-12-09 by the [reprex package](https://reprex.tidyverse.org) (v0.3.0)</sup>

# Ways that this data can be used:

* Study behavioural differences between mock and actual care
* Use in infection risk models to evaluate the route of fomite transmission
* Study behaviour of healthcare staff through Markov chains or other methods
