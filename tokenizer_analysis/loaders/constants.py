"""Constants for the multilingual data loaders."""

ISO639_1_to_ISO639_2 = {
    'en': 'eng',
    'es': 'spa',
    'de': 'deu',
    'ru': 'rus',
    'pl': 'pol',
    'it': 'ita',
    'fr': 'fra',
    'pt': 'por',
    'hu': 'hun',
    'fi': 'fin',
    'sv': 'swe',
    'cs': 'ces',
    'ca': 'cat',
    'mn': 'mon',
    'sh': 'hbs'
}

ISO639_1_to_FLORES = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'ar': 'arb_Arab',
    'cy': 'cym_Latn',
    'de': 'deu_Latn',
    'zh': 'cmn_Hans',  # Mandarin Chinese in Simplified script
    'ru': 'rus_Cyrl',
    'hi': 'hin_Deva',
    'sw': 'swh_Latn',
    'yo': 'yor_Latn',
    'el': 'ell_Grek',
    'gn': 'grn_Latn',
    'id': 'ind_Latn',
    'ku': 'kmr_Latn',  # Central Kurdish
    'mg': 'plt_Latn',  # Malagasy Plateau dialect
    'pl': 'pol_Latn',
    'tr': 'tur_Latn',
    'vi': 'vie_Latn',
    'ga': 'gle_Latn'
}
FLORES_to_ISO639_2 = {
    'eng_Latn': 'eng',
    'spa_Latn': 'spa',
    'deu_Latn': 'deu',
    'fra_Latn': 'fra',
    'rus_Cyrl': 'rus',
    'ita_Latn': 'ita',
    'por_Latn': 'por',
    'pol_Latn': 'pol',
    'jpn_Jpan': 'jpn',
    'vie_Latn': 'vie',
    'tur_Latn': 'tur',
    'nld_Latn': 'nld',
    'ind_Latn': 'ind',
    'arb_Arab': 'ara',
    'ces_Latn': 'ces',
    'pes_Arab': 'fas',  # Persian (modern) — ISO 639-2 code is 'fas'
    'fas_Arab': 'fas',
    'ell_Grek': 'ell',
    'cmn_Hani': 'zho',  # Mandarin Chinese — ISO 639-2 for Chinese is 'zho'
    'cmn_Hans': 'zho',  # Mandarin Chinese — ISO 639-2 for Chinese is 'zho'
    'hin_Deva': 'hin',
    'kor_Hang': 'kor',
    'tha_Thai': 'tha',
    'heb_Hebr': 'heb',
    'ben_Beng': 'ben',
    'tam_Taml': 'tam',
    'kat_Geor': 'kat',
    'mar_Deva': 'mar',
    'tel_Telu': 'tel',
    'nob_Latn': 'nob',  # Norwegian Bokmål
    'azj_Latn': 'aze',  # Azerbaijani
    'swe_Latn': 'swe',
    'ron_Latn': 'ron',
    'ukr_Cyrl': 'ukr',
    'hun_Latn': 'hun',
    'dan_Latn': 'dan',
    'fin_Latn': 'fin',
    'bul_Cyrl': 'bul',
    'slk_Latn': 'slk',
    'cat_Latn': 'cat',
    'zsm_Latn': 'zsm',  # Malay (modern standardized variety); may map to 'msa'
    'urd_Arab': 'urd',
    'bel_Cyrl': 'bel',
    'eus_Latn': 'eus',
    'tgk_Cyrl': 'tgk',
    'sot_Latn': 'sot',
    'yor_Latn': 'yor',
    'swh_Latn': 'swa',
    'ekk_Latn': 'est',
    'lvs_Latn': 'lav',
    'glg_Latn': 'glg',
    'cym_Latn': 'cym',
    'als_Latn': 'als',  # Albanian (Gheg), unofficial; ISO 639-2 for Albanian is 'alb' or 'sqi'
    'mkd_Cyrl': 'mkd',
    'mal_Mlym': 'mal',
    'mya_Mymr': 'mya',
    'guj_Gujr': 'guj',
    'afr_Latn': 'afr',
    'fil_Latn': 'fil',
    'haw_latn': 'haw',
    'uzn_Latn': 'uzb'}