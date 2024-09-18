library(caret)
library(cowplot)
library(cvms)
library(dplyr)
library(ggplot2)
library(forcats)
library(patchwork)
library(pheatmap)
library(randomizr)
library(RColorBrewer)
library(stringr)
library(viridis)
library(glue)
library(tidyr)

# Select model for analysis

model <- "20230426_082517"
dir.create(file.path("analysis/results/", model), showWarnings = FALSE)

# Load core model code results
# Note that "Anhinga" and "Unknown White" are removed from this data by the
# core modeling code

matched_data <- read.csv(glue("analysis/data/{model}/iou_dataframe.csv")) |>
  mutate(
    predicted_label = case_match(
      predicted_label,
      "" ~ "Bird Not Detected",
      .default = predicted_label)
  )

# Load observed and predicted count data
# In contrast to matched_data this data includes all predictions not just those
# associated with a true bird

sp_observations <- read.csv(glue("analysis/data/{model}/species_test.csv"))
sp_predictions <- read.csv(glue("analysis/data/{model}/predictions_dataframe.csv"))
sp_observations <- filter(sp_observations, label != "Anhinga")

# Total bird count 1:1 plot

bird_observation_counts <- sp_observations |>
  group_by(image_path) |>
  summarize(count = n())

bird_detector_counts <- sp_predictions |>
  group_by(image_path) |>
  summarize(count = n())

bird_counts <- full_join(bird_observation_counts,
                         bird_detector_counts,
                         by = "image_path") |>
                 mutate(observations = count.x, predictions = count.y)

bird_counts_plot <- ggplot(data = bird_counts,
                          aes(x = predictions, y = observations)) +
  geom_abline(slope = 1, intercept = 0) +
  geom_point(size = 3, pch=21, alpha = 0.75, color = "black", fill = "grey") +
  scale_y_sqrt(limits = c(0, 350), breaks = c(20, 50, 100, 200, 300)) +
  scale_x_sqrt(limits = c(0, 350), breaks = c(20, 50, 100, 200, 300)) +
  theme_bw(base_size = 16) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank()) +
  ylab("Human Counts") +
  xlab("Bird Detector")

ggsave(glue("analysis/results/{model}/bird_counts.png"), height = 5, width = 5)

## Precision & recall for bird detection



## Precision & Recall for Species model

# Boxes with overlap < 0.4 are considered not to match and therefore
# are excluded from species classification evaluation by convention.
# The failure to detect these birds is incorporated in the detection
# level precision and recall metrics.

matched_no_missed <- matched_data |>
  mutate(predicted_label = case_when(IoU < 0.4 ~ "Bird Not Detected",
                             .default = predicted_label)) |>
  filter(predicted_label != "Bird Not Detected")

species_list <- unique(matched_no_missed$true_label)
class_results <- data.frame(species = character(length(species_list)),
                  f1 = numeric(length(species_list)),
                  precision = numeric(length(species_list)),
                  recall = numeric(length(species_list))
                  )
for (i in seq_along(species_list)){
  species <- species_list[i]
  tp <- nrow(filter(matched_no_missed, true_label == species,
                   predicted_label == species))
  fp <- nrow(filter(matched_no_missed, true_label != species,
                   predicted_label == species))
  tn <- nrow(filter(matched_no_missed, true_label != species,
                   predicted_label != species))
  fn <- nrow(filter(matched_no_missed, true_label == species,
                   predicted_label != species))
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1 <- 2 * (precision * recall) / (precision + recall)
  class_results$species[i] <- species
  class_results$precision[i] <- precision
  class_results$recall[i] <- recall
  class_results$f1[i] <- f1
}

print(class_results)

## Bird count comparison by species

# Using sp_observations & sp_predictions causes issues because
# of the presence of "Unknown White" labels in sp_observations
# but shows decent counts for WHIBs. We can't just drop
# "Unknown White" birds from these tables because they aren't
# matched with the birds in sp_predictions

# Using matched_data (iou_dataframe.csv) properly excludes "Unknown White"
# birds, but it exhibits poorer performance for WHIBs when manually
# implementing the IoU threshold cut off.

# Together this suggests that sp_predictions may not be
# respecting the IoU threshold and including anything with
# >0 overlap, like matched_data was doing.

sp_observation_counts <- matched_data |>
  group_by(image_path, true_label) |>
  summarize(count = n()) |>
  rename(label = true_label) |>
  ungroup() |>
  complete(image_path, label, fill = list(count = 0))

bird_detector_sp_counts <- matched_data |>
  filter(predicted_label != "Bird Not Detected") |>
  group_by(image_path, predicted_label) |>
  summarize(count = n()) |>
  rename(label = predicted_label) |>
  ungroup() |>
  complete(image_path, label, fill = list(count = 0))

bird_counts <- full_join(sp_observation_counts,
                         bird_detector_sp_counts,
                         by = c("image_path", "label")) |>
                 rename(observations = count.x, predictions = count.y) |>
                 replace_na(list(observations = 0, predictions = 0))

# Include square root transform for overall relationship to balance contribution of
# high and low counts
f <- bird_counts$predictions
f_sqrt <- sqrt(f)
y <- bird_counts$observations
y_sqrt <- sqrt(y)

r2_1to1 <- 1 - sum((y - f)^2) / sum((y - mean(y))^2)
r2_1to1_sqrt <- 1 - sum((y_sqrt - f_sqrt)^2) / sum((y_sqrt - mean(y_sqrt))^2)
print("R^2 total:")
print(r2_1to1)
print("\n")
print("R^2 total sqrt transformed:")
print(r2_1to1_sqrt)

# Within species counts are more consistent to calculate R^2 on
# untransformed data
r2_1to1_sp <- bird_counts |>
  group_by(label) |>
  summarize(R2 = 1 - sum((observations - predictions)^2) /
    sum((observations - mean(observations))^2),
    RMSE = sqrt(mean((observations - predictions)^2)))

print("R^2 species:")
print(r2_1to1_sp)

bird_counts_no_labels <- bird_counts |>
  filter(label != "Empty") |>
  select(-label)

sp_counts <- ggplot(data = filter(bird_counts, label != "Empty"),
                    aes(x = predictions, y = observations)) +
  geom_abline(slope = 1, intercept = 0) +
  geom_point(data = bird_counts_no_labels, size = 1, color = "grey") +
  geom_point(size = 2, pch = 21, alpha = 0.5, fill = "black") +
  facet_wrap(~label) +
  scale_y_sqrt(limits = c(0, 350)) +
  scale_x_sqrt(limits = c(0, 350)) +
  theme_bw(base_size = 16) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank()) +
  ylab("Human Counts") +
  xlab("Bird Detector")

make_sp_count_zoom <- function(bird_counts, species){
  bird_counts_focal <- filter(bird_counts, label == species)
  ggplot(data = bird_counts_focal,
         mapping = aes(x = predictions, y = observations)) +
    geom_abline(slope = 1, intercept = 0) +
    geom_point(size = 1, pch = 21, alpha = 0.5, fill = "black") +
    xlim(min(bird_counts_focal$observations,
             bird_counts_focal$predictions) - 1,
         max(bird_counts_focal$observations,
             bird_counts_focal$predictions) + 1) +
    ylim(min(bird_counts_focal$observations,
             bird_counts_focal$predictions) - 1,
         max(bird_counts_focal$observations,
             bird_counts_focal$predictions) + 1) +
    theme_bw() +
    theme(axis.ticks.y = element_blank(),
          axis.text.y = element_blank(),
          axis.title.x = element_blank(),
          axis.ticks.x = element_blank(),
          axis.text.x = element_blank(),
          axis.title.y = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.minor.x = element_blank())
}

gbh_zoom <- make_sp_count_zoom(bird_counts, "Great Blue Heron")
greg_zoom <- make_sp_count_zoom(bird_counts, "Great Egret")
rosp_zoom <- make_sp_count_zoom(bird_counts, "Roseate Spoonbill")
sneg_zoom <- make_sp_count_zoom(bird_counts, "Snowy Egret")
whib_zoom <- make_sp_count_zoom(bird_counts, "White Ibis")
wost_zoom <- make_sp_count_zoom(bird_counts, "Wood Stork")
ggdraw(sp_counts) +
  draw_plot(gbh_zoom, 0.1, 0.75, 0.15, 0.15) +
  draw_plot(greg_zoom, 0.4, 0.75, 0.15, 0.15) +
  draw_plot(rosp_zoom, 0.7, 0.75, 0.15, 0.15) +
  draw_plot(sneg_zoom, 0.1, 0.32, 0.15, 0.15) +
  draw_plot(whib_zoom, 0.4, 0.32, 0.15, 0.15) +
  draw_plot(wost_zoom, 0.7, 0.32, 0.15, 0.15)

ggsave(glue("analysis/results/{model}/sp_counts.png"), height = 5, width = 8)

ggplot(data = filter(bird_counts, label != "Empty"),
       aes(x = predictions, y = observations)) +
  geom_abline(slope = 1, intercept = 0) +
  geom_point(size = 3, pch=21, alpha = 0.5, color = "black", fill = "grey") +
  facet_wrap(~label, scales="free") +
  theme_bw(base_size = 16) +
  theme(panel.grid.major.y=element_blank(),
        panel.grid.major.x=element_blank(),
        panel.grid.minor.y=element_blank(),
        panel.grid.minor.x=element_blank()) +
  ylab("Human Counts") +
  xlab("Bird Detector")

ggsave(glue("analysis/results/{model}/sp_counts_zoom.png"),
  height = 5, width = 8)


# Aggregated test count data

# Test tiles are typically much smaller than colonies, so aggregate to get a
# better feel for performance at largers scales

set.seed(26)
image_path <- unique(bird_counts$image_path)
num_images <- length(image_path)
num_tiles <- 5
num_groups <- num_images %/% num_tiles

group <- complete_ra(N = num_images, num_arms = num_groups)
group_data <- data.frame(image_path, group)

bird_counts_grouped <- bird_counts |>
  full_join(group_data, by = c("image_path")) |>
  group_by(group, label) |>
  summarize(observations = sum(observations),
            predictions = sum(predictions)) |>
  filter(label != "Empty")

bird_counts_group_no_label = bird_counts_grouped |>
  select(-label)

birds_per_image = bird_counts_grouped |>
  group_by(group) |>
  summarize(count = sum(observations))

ggplot(birds_per_image, aes(x = count)) +
  geom_density()

sp_counts_grouped = ggplot(filter(bird_counts_grouped, label != "Empty"),
                           aes(x = predictions, y = observations)) +
  geom_abline(slope = 1, intercept = 0) +
  geom_point(data = bird_counts_group_no_label, size = 1, color = "grey") +
  geom_point(size = 2, pch = 21, alpha = 0.5, fill = "black") +
  facet_wrap(~label) +
  scale_y_sqrt(limits = c(0, 600)) +
  scale_x_sqrt(limits = c(0, 600)) +
  theme_bw(base_size = 16) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank()) +
  ylab("Human Counts") +
  xlab("Bird Detector")

gbh_zoom <- make_sp_count_zoom(bird_counts_grouped, "Great Blue Heron")
greg_zoom <- make_sp_count_zoom(bird_counts_grouped, "Great Egret")
rosp_zoom <- make_sp_count_zoom(bird_counts_grouped, "Roseate Spoonbill")
sneg_zoom <- make_sp_count_zoom(bird_counts_grouped, "Snowy Egret")
whib_zoom <- make_sp_count_zoom(bird_counts_grouped, "White Ibis")
wost_zoom <- make_sp_count_zoom(bird_counts_grouped, "Wood Stork")
ggdraw(sp_counts_grouped) +
  draw_plot(gbh_zoom, 0.1, 0.75, 0.15, 0.15) +
  draw_plot(greg_zoom, 0.4, 0.75, 0.15, 0.15) +
  draw_plot(rosp_zoom, 0.7, 0.75, 0.15, 0.15) +
  draw_plot(sneg_zoom, 0.1, 0.32, 0.15, 0.15) +
  draw_plot(whib_zoom, 0.4, 0.32, 0.15, 0.15) +
  draw_plot(wost_zoom, 0.7, 0.32, 0.15, 0.15)

ggsave(glue("analysis/results/{model}/sp_counts_5_tiles.png"),
  height = 5, width = 8)

## Confusion matrices

make_confusion_matrix <- function(conf_mat, label_order_target, label_order_prediction) {
  conf_mat_df <- conf_mat$Table[[1]]
  conf_mat_df <- conf_mat_df |>
    as.data.frame(stringsAsFactors = TRUE) |>
    group_by(Target) |>
    mutate(Count = Freq,
          Percent = Freq / sum(Freq) * 100,
          Proportion = Freq / sum(Freq),
          Target = factor(Target, rev(label_order_target)),
          Prediction = factor(Prediction, label_order_prediction)) |>
    ungroup() |>
    filter(Target != "Bird Not Detected") |>
    select(-Freq)

  ggplot(conf_mat_df, aes(x = Prediction, y = Target, fill = Percent)) +
    geom_tile() +
    coord_equal() +
    geom_text(aes(label = paste(scales::percent(Proportion, accuracy = 0.1),
                            "\n", "(", Count, ")"))) +
    scale_fill_gradient(low = "white", high = "#3575b5") +
    scale_x_discrete(labels = str_replace_all(label_order_prediction,
                                " ", "\n")) +
    scale_y_discrete(labels = str_replace_all(rev(label_order_target),
                                " ", "\n")) +
    labs(x = "Predicted", y = "Observed") +
    theme_bw(base_size = 14) +
    theme(plot.title = element_text(size = 25, hjust = 0.5,
                                    margin = margin(20, 0, 20, 0)),
          legend.title = element_text(size = 14, margin = margin(0, 20, 10, 0)),
          axis.title.x = element_text(margin = margin(20, 20, 20, 20), size = 18),
          axis.title.y = element_text(margin = margin(0, 20, 0, 10), size = 18),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.position = "none")
}

# Boxes with IoU's below 0.4 are considered to be missed
matched_iou_threshed <- matched_data |>
  mutate(predicted_label = case_when(IoU < 0.4 ~ "Bird Not Detected",
                             .default = predicted_label))

# Order the labels by commonness in test
sp_counts_ordered <- matched_iou_threshed |>
  group_by(true_label) |>
  summarize(count = n()) |>
  arrange(desc(count))
label_order_target <- sp_counts_ordered$true_label
label_order_prediction <- c(label_order_target, "Bird Not Detected")

# Standard confusion matrix
# Version of confusion matrix that focuses only species classification task

conf_mat_standard <- confusion_matrix(
  targets = matched_no_missed$true_label,
  predictions = matched_no_missed$predicted_label
)

make_confusion_matrix(
  conf_mat_standard,
  label_order_target,
  label_order_prediction
)

ggsave(glue("analysis/results/{model}/confusion_matrix_standard.png"),
  height = 8, width = 8)

# Prediction confusion matrix
# Version of confusion matrix that includes cases where birds are not detected
conf_mat_prediction <- confusion_matrix(
  targets = matched_iou_threshed$true_label,
  predictions = matched_iou_threshed$predicted_label)

make_confusion_matrix(
  conf_mat_prediction,
  label_order_target,
  label_order_prediction
)

ggsave(glue("analysis/results/{model}/confusion_matrix_prediction.png"),
  height = 8, width = 8)
