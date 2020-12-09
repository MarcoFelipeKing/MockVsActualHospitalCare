# MockVsActualHospitalCare
Data behind the journal article: Why is mock care not a good proxy for predicting hand contamination during patient care? (https://doi.org/10.1016/j.jhin.2020.11.016)

These data are sequential surface contacts during mock patient care and actual hospital care for three types of care: Taking Observations and vitals, IV drip manipulation and Drs/MD rounds.

# Headers in the .csv file
ActivityID	CareType	HCWType	RoomType	Surface	Date	Time	Dev.Date.Time	type

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
