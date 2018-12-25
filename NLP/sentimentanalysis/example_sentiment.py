# encoding: utf-8

"""
Here we are exploring the posibilities of sentiment detection...

Sentiment detection is of course nothing more than just classification of two classes... The key thing here is of
course on how to build the features. Because we assume that the words used in a sentence are important


"""

# In order to start the CoreNLP server... follow all instructions mentioned here
#
# Then move to your Powershell (or cmd) and execute the following like
#
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
#
# Don't worry that you don't see edu.stanford.nlp.pipeline.StanfordCoreNLPServer in the directory..
#
# timeout is set in miliseconds... so the server will be up for quite a short time now.
# Now we have set up the server at http://localhost:9000


from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("I love you. I hate him. You are nice. He is dumb",
                   properties={
                       'annotators': 'sentiment',
                       'outputFormat': 'json',
                       'timeout': 1000,
                   })
for s in res["sentences"]:
    print("%d: '%s': %s %s" % (
        s["index"],
        " ".join([t["word"] for t in s["tokens"]]),
        s["sentimentValue"], s["sentiment"]))


