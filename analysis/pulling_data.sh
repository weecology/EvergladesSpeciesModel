MODEL=20230426_082517
mkdir analysis/data/$MODEL/
scp ethanwhite@hpg.rc.ufl.edu:/blue/ewhite/everglades/Zooniverse/$MODEL/'*'.csv analysis/data/$MODEL/
scp ethanwhite@hpg.rc.ufl.edu:/blue/ewhite/everglades/Zooniverse/cleaned_test/test_resized.csv analysis/data/$MODEL/species_test.csv