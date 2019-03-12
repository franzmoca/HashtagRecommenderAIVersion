#!/bin/bash
sbt --error assembly &&
spark-submit \
  --class "Main" \
--driver-memory 2g \
--executor-cores 6 \
--executor-memory 10g \
  ./target/scala-2.11/scalable-hashtag-recommender-system-assembly-0.1.jar \
--features-file /Users/vincenzo/Downloads/pca130.csv --tag-file /Users/vincenzo/iaproject/dataset/tag_listc.txt --image-path "https://image.ibb.co/kYdbKT/IMG_20180725_194058_490.jpg" --out-cluster-file clusters.csv

