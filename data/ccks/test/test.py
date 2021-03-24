test_tags = open("tags.txt")
test_sens = open("sentences.txt")

tags, sens = [], []
for line in test_tags.readlines():
#     print(line)
    tags.append(line)

for line in test_sens.readlines():
#     print(line)
    sens.append(line)

# print(len(tags))
# print(len(sens))

assert len(tags) == len(sens)

for i in range(len(sens)):
    assert len(tags[i]) == len(sens[i])
print("Done")

sentences, tags = [], []
with open("sentences.txt", 'r') as file:
    for line in file:
        tokens = line.strip()
        sentences.append(tokens.split(" "))
# replace each tag by its index
with open("tags.txt", 'r') as file:
    for line in file:
        tag_seq = [tag for tag in line.strip().split(' ')]
        tags.append(tag_seq)

# checks to ensure there is a tag for each token
assert len(sentences) == len(tags)
for s in range(len(sentences)):
    # print(s)
    if len(tags[s]) != len(sentences[s]):
        print(tags[s])
        print(sentences[s])
        print(len(tags[s]))
        print(len(sentences[s]))
    assert len(tags[s]) == len(sentences[s])
