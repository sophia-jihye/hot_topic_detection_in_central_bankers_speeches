#!/usr/bin/env python
# coding: utf-8

# AdaptKeyBERT: https://github.com/AmanPriyanshu/AdaptKeyBERT
# 
# ```bash
# pip install adaptkeybert
# ```

from adaptkeybert import KeyBERT
import os
import pandas as pd
from tqdm import tqdm

filepath = 'data/preprocessed_sentences_for_each_period_1997Q1-2019Q3.csv'

save_dir = 'C:\DATA\hot_topic_detection_in_central_bankers_speeches'
save_filepath = os.path.join(save_dir, 'top5_hot_topics_unigram_AdaptKeyBERT.csv')

def normalize_bigrams(original_df):
    original_df.sort_values(by=['score'], ascending=False, inplace=True)
    original_df['sorted_word'] = original_df['word'].apply(lambda x: sorted(x.split()))
    
    no_duplicated_df = original_df.drop_duplicates(subset=['sorted_word']).drop(columns=['sorted_word'])
    
    word_score_dict = dict(zip(original_df['word'].values, original_df['score'].values))
    def sum_scores(word):
        tokens = word.split()
        return word_score_dict.setdefault(' '.join(sorted(tokens, reverse=False)), 0) \
                    + word_score_dict.setdefault(' '.join(sorted(tokens, reverse=True)), 0)

    if len(original_df) != len(no_duplicated_df):
        no_duplicated_df['score'] = no_duplicated_df.apply(lambda x: sum_scores(x['word']), axis=1)
        
    return no_duplicated_df

if __name__ == '__main__':
    
    df = pd.read_csv(filepath)
    sorted_periods = sorted(df[df['period'].apply(lambda x: x[:4]!='1997')].period.unique())
    df.set_index('period', inplace=True)
    
    dfs = []
    for period in tqdm(sorted_periods):
        doc = df.loc[period]['document']

        kw_model = KeyBERT(domain_adapt=True, zero_adapt=True)
        kw_model.pre_train([doc], [['supervised', 'unsupervised']], lr=1e-3)
        kw_model.zeroshot_pre_train(['supervised', 'unsupervised'], adaptive_thr=0.15)
        keywords = kw_model.extract_keywords(doc, top_n=5) # keyphrase_ngram_range=(2, 2) does not work.
        one_period_df = pd.DataFrame(keywords, columns=['word', 'score'])

        one_period_df = normalize_bigrams(one_period_df)

        one_period_df['period'] = period
        dfs.append(one_period_df)
    top5_df = pd.concat(dfs)

    top5_df.to_csv(save_filepath, index=False)
    print('Created {}'.format(save_filepath))

    # Key periods
    print(top5_df[(top5_df['period'].isin(['1998_Q2', '2000_Q1', '2007_Q2']))])

