"""
The UDify filenames have some odd naming conventions sometimes.
Used for automatically printing tables and iterating over all train data.
"""

no_train_set = ["UD_Breton-KEB", "UD_Faroese-OFT", "UD_Swedish-PUD"]
no_train_set_lowercase = ["br_keb-ud", "fo_oft-ud", "sv_pud-ud"]

my_order = [
    "UD_Armenian-ArmTDP",
    "UD_Breton-KEB",
    "UD_Buryat-BDT",
    "UD_Faroese-OFT",
    "UD_Kazakh-KTB",
    "UD_Upper_Sorbian-UFAL",
    "UD_Arabic-PADT",
    "UD_Czech-PDT",
    "UD_English-EWT",
    "UD_Finnish-TDT",
    "UD_French-Spoken",
    "UD_German-GSD",
    "UD_Hindi-HDTB",
    "UD_Hungarian-Szeged",
    "UD_Italian-ISDT",
    "UD_Japanese-GSD",
    "UD_Korean-Kaist",
    "UD_Norwegian-Nynorsk",
    "UD_Persian-Seraji",
    "UD_Russian-SynTagRus",
    "UD_Swedish-PUD",
    "UD_Tamil-TTB",
    "UD_Urdu-UDTB",
    "UD_Vietnamese-VTB",
    "UD_Bulgarian-BTB",
    "UD_Telugu-MTG",
]

my_order2 = [
    "UD_Armenian-ArmTDP",
    "UD_Breton-KEB",
    "UD_Buryat-BDT",
    "UD_Faroese-OFT",
    "UD_Kazakh-KTB",
    "UD_Upper_Sorbian-UFAL",
    "UD_Finnish-TDT",
    "UD_French-Spoken",
    "UD_German-GSD",
    "UD_Hungarian-Szeged",
    "UD_Japanese-GSD",
    "UD_Persian-Seraji",
    "UD_Swedish-PUD",
    "UD_Tamil-TTB",
    "UD_Urdu-UDTB",
    "UD_Vietnamese-VTB",
    "UD_Bulgarian-BTB",
    "UD_Telugu-MTG",
    "UD_Arabic-PADT",
    "UD_Czech-PDT",
    "UD_English-EWT",
    "UD_Hindi-HDTB",
    "UD_Italian-ISDT",
    "UD_Korean-Kaist",
    "UD_Norwegian-Nynorsk",
    "UD_Russian-SynTagRus",
]

languages = [
    "UD_Breton-KEB",
    "UD_Arabic-PADT",
    "UD_Armenian-ArmTDP",
    "UD_Bulgarian-BTB",
    "UD_Buryat-BDT",
    "UD_Czech-PDT",
    "UD_English-EWT",
    "UD_Faroese-OFT",
    "UD_Finnish-TDT",
    "UD_French-Spoken",
    "UD_German-GSD",
    "UD_Hindi-HDTB",
    "UD_Hungarian-Szeged",
    "UD_Italian-ISDT",
    "UD_Japanese-GSD",
    "UD_Kazakh-KTB",
    "UD_Korean-Kaist",
    "UD_Norwegian-Nynorsk",
    "UD_Persian-Seraji",
    "UD_Russian-SynTagRus",
    "UD_Swedish-PUD",
    "UD_Tamil-TTB",
    "UD_Telugu-MTG",
    "UD_Upper_Sorbian-UFAL",
    "UD_Urdu-UDTB",
    "UD_Vietnamese-VTB",
]

languages_too_small_for_20_batch_20 = [
    "UD_Breton-KEB",
    "UD_Buryat-BDT",
    "UD_Faroese-OFT",
    "UD_Kazakh-KTB",
    "UD_Swedish-PUD",
    "UD_Upper_Sorbian-UFAL",
]

languages_too_small_for_20_batch_20_lowercase = [
    "br_keb-ud",
    "bxr_bdt-ud",
    "fo_oft-ud",
    "kk_ktb-ud",
    "sv_pud-ud",
    "hsb_ufal-ud",
]

train_languages_readable = [
    "Arabic",
    "Czech",
    "English",
    "Hindi",
    "Italian",
    "Korean",
    "Norwegian",
    "Russian",
]

languages_readable = [
    "Arabic",
    "Armenian",
    "Breton",
    "Bulgarian",
    "Buryat",
    "Czech",
    "English",
    "Faroese",
    "Finnish",
    "French",
    "German",
    "Hindi",
    "Hungarian",
    "Italian",
    "Japanese",
    "Kazakh",
    "Korean",
    "Norwegian",
    "Persian",
    "Russian",
    "Swedish",
    "Tamil",
    "Telugu",
    "UpperSorbian",
    "Urdu",
    "Vietnamese",
]

languages_lowercase = [
    "br_keb-ud",
    "ar_padt-ud",
    "hy_armtdp-ud",
    "bg_btb-ud",
    "bxr_bdt-ud",
    "cs_pdt-ud",
    "en_ewt-ud",
    "fo_oft-ud",
    "fi_tdt-ud",
    "fr_spoken-ud",
    "de_gsd-ud",
    "hi_hdtb-ud",
    "hu_szeged-ud",
    "it_isdt-ud",
    "ja_gsd-ud",
    "kk_ktb-ud",
    "ko_kaist-ud",
    "no_nynorsk-ud",
    "fa_seraji-ud",
    "ru_syntagrus-ud",
    "sv_pud-ud",
    "ta_ttb-ud",
    "te_mtg-ud",
    "hsb_ufal-ud",
    "ur_udtb-ud",
    "vi_vtb-ud",
]
train_languages = [
    "UD_Arabic-PADT",
    "UD_Czech-PDT",
    "UD_Italian-ISDT",
    "UD_Norwegian-Nynorsk",
    "UD_Russian-SynTagRus",
    "UD_Hindi-HDTB",
    "UD_Korean-Kaist",
]

train_languages_lowercase = [
    "ar_padt-ud",
    "cs_pdt-ud",
    "it_isdt-ud",
    "no_nynorsk-ud",
    "ru_syntagrus-ud",
    "hi_hdtb-ud",
    "ko_kaist-ud",
]

validation_languages = ["UD_Bulgarian-BTB", "UD_Telugu-MTG"]

validation_languages_lowercase = ["bg_btb-ud", "te_mtg-ud"]
