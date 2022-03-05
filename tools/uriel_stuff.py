from scipy import spatial
from collections import defaultdict
import lang2vec.lang2vec as l2v


lang2iso = {
    "Arabic": "arb",
    "Armenian": "hye",
    "Breton": "bre",
    "Buryat": "bxm",
    "Bulgarian": "bul",
    "Czech": "ces",
    "English": "eng",
    "Faroese": "fao",
    "Finnish": "fin",
    "French": "fra",
    "German": "deu",
    "Hindi": "hin",
    "Hungarian": "hun",
    "Italian": "ita",
    "Japanese": "jpn",
    "Kazakh": "kaz",
    "Korean": "kor",
    "Norwegian": "nor",
    "Persian": "pes",
    "Russian": "rus",
    "Swedish": "swe",
    "Tamil": "tam",
    "Telugu": "tel",
    "UpperSorbian": "hsb",
    "Urdu": "urd",
    "Vietnamese": "vie",
}

feature_names = [
    "syntax_knn",
    "phonology_knn",
    "inventory_knn",
    "fam",  # Membership in language families and subfamilies
    "geo",  # Distance from fixed points on Earth's surface
]

iso2lang = {v: k for k, v in lang2iso.items()}

testset = lang2iso.values()


def generate_distances(feature_name):
    """
    Generates the distances for a specific URIEL feature
    (defined in feature_names).

    Input:
    ---------------
    feature_name: str (from feature_names)

    Output:
    feature_data: feature_data[lan1][lan2] = cosine distance
    language_data: binary vector for each language
    """
    language_data = defaultdict(list)
    feature_data = defaultdict(lambda: defaultdict(lambda: 0))
    for compare_lang in testset:
        features = l2v.get_features(list(lang2iso.values()), feature_name)
        a = features[compare_lang]
        language_data[compare_lang].append(a)
        for lang, iso in lang2iso.items():
            b = features[iso]
            c, d = [], []
            for x, y in zip(a, b):
                if not (x == "--" or y == "--"):
                    c.append(x)
                    d.append(y)
            distance = 1 - spatial.distance.cosine(c, d)
            feature_data[lang][iso2lang[compare_lang]] = distance
    return feature_data, language_data


cosines, binaryvecs = generate_distances(feature_name="syntax_knn")

