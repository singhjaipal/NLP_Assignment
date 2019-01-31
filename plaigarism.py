
import glob, nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('punkt') # if necessary...
stemmer = nltk.stem.porter.PorterStemmer()
rpm = dict((ord(char), None) for char in string.punctuation)
def st(tokens):
    return [stemmer.stem(item) for item in tokens]
def n(text):
    return st(nltk.word_tokenize(text.lower().translate(rpm)))
tfv = TfidfVectorizer(analyzer=n, min_df=0, stop_words='english', sublinear_tf=True)
inp = str(input())
files = sorted(glob.glob(inp + "/*.c"))
files += sorted(glob.glob("R/*.c"))
corpus = [open(file, encoding="utf8").read() for file in files]

tfm = tfv.fit_transform(corpus)
sm = (tfm*tfm.T).A
for x in range(0, len(corpus)):
    for y in range(x+1, len(corpus)):
        if sm[x,y] > 0.5:
            print(f"{files[x]}, {files[y]} plagarized {round(sm[x,y]*100,2)}%")
