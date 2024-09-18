library(dplyr)
library(glue)
library(readr)

model <- "20230426_082517"

train_data <- read_csv(glue("analysis/data/{model}/species_train.csv"))
train_data_full <- read_csv(glue("analysis/data/{model}/species_train_full.csv"))
test_data <- read_csv(glue("analysis/data/{model}/species_test.csv"))

zooniverse_data <- train_data_full |>
  filter(grepl("[0-9]{8}.png", image_path))
weakly_labeled_data <- train_data_full |>
  filter(!grepl("[0-9]{8}.png", image_path)) |>
  filter(xmin != 0, xmax != 0, ymin != 0, ymax != 0)
empty_frame_data <- train_data_full |>
  filter(xmin == 0, xmax == 0, ymin == 0, ymax == 0)

#Zooniverse train crops
crop_count_train <- train_data |>
  distinct(image_path) |>
  summarize(count = n())
crop_count_train

#Bounding box delineations
crop_count_bb_train <- train_data |>
  filter((xmax - xmin != 50) & (ymax - ymin != 50)) |>
  filter((xmax - xmin !=36) & (ymax - ymin != 36)) |>
  distinct(image_path) |>
  summarize(count = n())
crop_count_bb_train
  
#Empty train crops
crop_count_empty_frames <- train_data_full |>
  filter(xmin == 0, xmax == 0, ymin == 0, ymax == 0) |>
  distinct(image_path) |>
  nrow()
crop_count_empty_frames

#Weakly labeled crops
crop_count_train_all <- train_data_full |>
  distinct(image_path) |>
  summarize(count = n())
crop_count_weakly_labeled <- crop_count_train_all - crop_count_train - crop_count_empty_frames
crop_count_weakly_labeled

#Test crops
crop_count_test <- test_data |>
  distinct(image_path) |>
  summarize(count = n())
crop_count_test

crop_count_total <- crop_count_train + crop_count_test
crop_count_total

#Bird count - zooniverse
bird_count_train <- nrow(train_data)
bird_count_train

#Bird count - weakly labeled
bird_count_train_all <- nrow(train_data_full)
bird_count_weakly_labeled <- bird_count_train_all - bird_count_train - crop_count_empty_frames
bird_count_weakly_labeled

#Bird count - test
bird_count_test <- nrow(test_data)
bird_count_test

bird_count_test_unknown <- filter(test_data, label == "Unknown White") |> nrow()
bird_count_test_unknown

