"""
Cross validation code here.

Mike Lumetta, Chris Miller
"""

from parseTweet import parse_tweets

def k_chunks(data, chunks):
    data_dict = data['tweets']
    results = [dict() for i in range(chunks)]
    for j in range(chunks):
        results[j]['tweets'] = {}
    i = 0
    for k in sorted(list(data_dict.keys())):
        results[i]['tweets'][k] = data_dict[k]
        j += 1
        if i < chunks-1:
            i += 1
        else:
            i = 0
    return results

def combine_dictionaries(args):
    result = {}
    for arg in args:
        for key in arg.keys():
            if key not in result:
                result[key] = arg[key]
            else:
                for subkey in arg[key].keys():
                    result[key][subkey] = arg[key][subkey]
    return result

if __name__ == "__main__":
    filename = '/data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv'
    tweetData = parse_tweets(filename, 'B')
    chunks = k_chunks(tweetData, 5)
    for i in range(len(chunks)):
        print("Chunk", i, "has", len(chunks[i]['tweets']), "instances")
    print("tweetData has length", len(tweetData['tweets']))
    print()

    superdict = combine_dictionaries([chunks[0], chunks[2], chunks[3]])
    print("Combined has", len(superdict['tweets']), "tweets")
