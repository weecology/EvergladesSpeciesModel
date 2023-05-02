import polars as pl

def resize_boxes(df, reduction = 7):
    """Reduce box dimensions for WHIBs & SNEGs by 2 * reduction
    
    E.g., a 50x50 pixel box, with reduction = 7, will produce a
    WHIB & SNEG boxes that are 36x36 pixels.
    """
    box_columns = ["xmin", "ymin", "xmax", "ymax"]
    resized_df = (
        df.with_columns(
        pl.when(((pl.col("xmax") - pl.col("xmin") == 50)
                 & (pl.col("ymax") - pl.col("ymin") == 50)
                 & ((pl.col("label") == "White Ibis") |
                    (pl.col("label") == "Snowy Egret"))))
        .then(pl.struct([
            (pl.col("xmin") + reduction).alias("xmin"),
            (pl.col("ymin") + reduction).alias("ymin"),
            (pl.col("xmax") - reduction).alias("xmax"),
            (pl.col("ymax") - reduction).alias("ymax")
        ]))
        .otherwise(pl.struct(box_columns))
        .alias('name_struct')
        )
        .drop(box_columns)
        .unnest('name_struct')
    )
    return(resized_df)

train_path = "/blue/ewhite/everglades/Zooniverse/parsed_images/species_train.csv"
test_path = "/blue/ewhite/everglades/Zooniverse/cleaned_test/test.csv"

train = pl.read_csv(train_path)
test = pl.read_csv(test_path)

resized_train = resize_boxes(train)
resized_test = resize_boxes(test)
resized_train.write_csv("/blue/ewhite/everglades/Zooniverse/parsed_images/species_train_resized.csv")
resized_test.write_csv("/blue/ewhite/everglades/Zooniverse/cleaned_test/test_resized.csv")

resized_test.filter(pl.col("label") == "Snowy Egret")