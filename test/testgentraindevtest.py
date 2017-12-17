from preprocess.tweettovect import GenerateTrainDevTestSets

print("Starting")
generator = GenerateTrainDevTestSets("labeled_tweets.csv", 24783)
generator.generate()
print("Done")