MODEL=20230426_082517
mkdir analysis/data/$MODEL/
scp ethanwhite@hpg.rc.ufl.edu:/blue/ewhite/everglades/Zooniverse/$MODEL/'*'.csv analysis/data/$MODEL/
scp ethanwhite@hpg.rc.ufl.edu:/blue/ewhite/everglades/Zooniverse/cleaned_test/test.csv analysis/data/$MODEL/species_test.csv
scp ethanwhite@hpg.rc.ufl.edu:/blue/ewhite/everglades/Zooniverse/parsed_images/species_train_resized.csv analysis/data/$MODEL/species_train.csv
scp ethanwhite@hpg.rc.ufl.edu:/blue/ewhite/everglades/Zooniverse/parsed_images/species_train_$MODEL.csv analysis/data/$MODEL/species_train_full.csv