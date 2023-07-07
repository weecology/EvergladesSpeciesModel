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
library(Hmisc)

# Select model for analysis

model <- "20230426_082517"
dir.create(file.path("analysis/results/", model), showWarnings = FALSE)

# Load core model code results
# Note that "Anhinga" and "Unknown White" are removed from this data by the
# core modeling code

matched_data <- read.csv(glue("analysis/data/{model}/iou_dataframe.csv")) |>
  mutate(
    matched_data,
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

## Precision & Recall for Species model

species_list = unique(matched_data$true_label)
results = data.frame(species = character(length(species_list)),
                     precision = numeric(length(species_list)),
                     recall = numeric(length(species_list)))
for (i in seq_along(species_list)){
  species <- species_list[i]
  tp <- nrow(filter(matched_data, matched_data$true_label == species,
                   matched_data$predicted_label == species))
  fp <- nrow(filter(matched_data, matched_data$true_label != species,
                   matched_data$predicted_label == species))
  tn <- nrow(filter(matched_data, matched_data$true_label != species,
                   matched_data$predicted_label != species))
  fn <- nrow(filter(matched_data, matched_data$true_label == species,
                   matched_data$predicted_label != species))
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  results$species[i] <- species
  results$precision[i] <- precision
  results$recall[i] <- recall
}

print(results)

## Bird count comparison by species

sp_observation_counts <- matched_data |>
  group_by(image_path, true_label) |>
  summarize(count = n()) |>
  rename(label = true_label)

bird_detector_sp_counts <- matched_data |>
  filter(predicted_label != "Bird Not Detected") |>
  group_by(image_path, predicted_label) |>
  summarize(count = n()) |>
  rename(label = predicted_label)

bird_counts <- full_join(sp_observation_counts,
                         bird_detector_sp_counts,
                         by = c("image_path", "label")) |>
                 rename(observations = count.x, predictions = count.y) |>
                 replace_na(list(observations = 0, predictions = 0))

f <- sqrt(bird_counts$predictions)
y <- sqrt(bird_counts$observations)

r2_1to1 <- 1 - sum((y - f)^2) / sum((y - mean(y))^2)

r2_1to1_sp <- bird_counts |>
  group_by(label) |>
  summarize(R2 = 1 - sum((sqrt(observations) - sqrt(predictions))^2) /
    sum((sqrt(observations) - mean(sqrt(observations)))^2))

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

### Confusion matrix

sp_counts_ordered = matched_data |>
  group_by(true_label) |>
  summarize(count = n()) |>
  arrange(desc(count))
label_order_target = sp_counts_ordered$true_label
label_order_prediction = c(label_order_target, "Bird Not Detected")
conf_mat <- confusion_matrix(targets=matched_data$true_label,
                            predictions=matched_data$predicted_label)
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

ggsave(glue("analysis/results/{model}/confusion_matrix.png"),
  height = 8, width = 8)

### Uncertainty

whib_data <- filter(matched_data,
                    true_label == "Wood Stork",
                    predicted_label != "Bird Not Detected")
whib_breaks <- quantile(whib_data$score, probs = seq(0, 1, 0.2), na.rm = TRUE)
uncertainty_by_score_macro = matched_data %>%
  group_by(true_label) %>%
  filter(predicted_label != "Bird Not Detected") %>%
  mutate(score_category=cut2(score, g = 5, levels.mean = TRUE)) %>%
  group_by(true_label, score_category) %>%
  dplyr::summarize(accuracy = sum(predicted_label == true_label) / n()) %>%
  mutate(score_category_num = as.numeric(as.character(score_category)))

ggplot(uncertainty_by_score_macro, aes(x = score_category_num, y = accuracy)) +
  geom_point(size = 3, pch=21, color = "black", fill = "grey") +
  geom_abline(slope = 1, intercept = 0) +
  ylim(0, 1) +
  xlim(0.3, 1) +
  facet_wrap(~ true_label) +
  theme_bw() +
  theme_bw(base_size = 16) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank())

ggsave(glue("analysis/results/{model}/species_uncertainty.png",
  height = 5, width = 8))
