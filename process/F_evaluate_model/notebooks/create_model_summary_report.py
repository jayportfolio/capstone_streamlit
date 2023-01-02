#!/usr/bin/env python
# coding: utf-8


import json
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

from bs4 import BeautifulSoup


def include_in_html_report(type, section_header=None, section_figure=None, section_content=None, section_content_list=None):

    results_root = '../../F_evaluate_model'
    writePath_html = f'{results_root}/html/{key}.html'.replace(" ", "_").replace("(", "_").replace(")", "_")
    writePath_md = f'{results_root}/markdown/{key}.md'

    if not section_content_list:
        section_content_list = [section_content]

    if type == 'header':
        w = 'w' if section_figure <= 1 else 'a'
        with open(writePath_html, w) as f1:
            headers = f'<h{section_figure}>{section_content}</h{section_figure}>'
            f1.write(headers)
        with open(writePath_md, w) as f2:
            headers = f'{"#" * int(section_figure)} {section_content }\n'
            f2.write(headers)
    else:
        if section_header:
            with open(writePath_html, 'a') as f1:
                f1.write(f'<h3>{section_header}</h3>')
            with open(writePath_md, 'a') as f2:
                f2.write(f'### {section_header}\n')

        if type=='dataframe':
            with open(writePath_html, 'a') as f1:
                dfAsString = section_content.to_html()
                f1.write(dfAsString)
            with open(writePath_md, 'a') as f2:
                dfAsString = section_content.to_markdown()
                f2.write(dfAsString + '\n')
        elif type=='graph':
            filename = key + "_" + section_content
            #section_figure.savefig(f'model_results/artifacts/{filename.replace(" ", "_")}')
            section_figure.savefig(f'{results_root}/artifacts/{filename.replace(" ", "_").replace("(", "_").replace(")", "_")}')

            with open(writePath_html, 'a') as f1:
                dfAsString = f'<img src="../artifacts/{filename.replace(" ","_").replace("(", "_").replace(")", "_")}"/>'
                f1.write(dfAsString)

            with open(writePath_md, 'a') as f2:
                #dfAsString = f'(./model_results/artifacts/{filename}) \n'
                #dfAsString = f'![detail](./artifacts/{filename.replace(" ","_")})'
                dfAsString = f'![detail](../artifacts/{filename.replace(" ","_").replace("(", "_").replace(")", "_")})'
                f2.write(dfAsString)
                f2.write('\n')
        elif type=='json':

            with open(writePath_html, 'a') as f1:
                #f.write(json.dumps(html_content_dictionary, indent=4))
                soup = BeautifulSoup(section_content, "html.parser")
                f1.write(str(soup.prettify()))
            with open(writePath_md, 'a') as f2:
                soup = BeautifulSoup(section_content, "html.parser")

                import ast
                loads = ast.literal_eval(section_content)
                for each in loads:
                    f2.write(each + " = " + str(loads[each]) + "\n\n")

        elif type=='dict':

            for section_content in section_content_list:
                if isinstance(section_content, str):
                    import ast
                    section_content = ast.literal_eval(section_content)

                with open(writePath_html, 'a') as f1:
                    soup = BeautifulSoup(str(section_content), "html.parser")
                    f1.write(str(soup.prettify()))
                with open(writePath_md, 'a') as f2:
                    for each in section_content:
                        f2.write(each + " = " + str(section_content[each]) + "\n\n")

        elif type=='text':
            with open(writePath_html, 'a') as f1:
                for each_line in section_content_list:
                    f1.write(each_line + '<br>')
            with open(writePath_md, 'a') as f2:
                for each_line in section_content_list:
                    f2.write(each_line + '\n\n')

        with open(writePath_html, 'a') as f1:
            f1.write('<hr>')


key = 'Summary'
include_in_html_report("header", section_content=f"SUMMARY RESULTS", section_figure=1)

end_timestamp = datetime.now()

include_in_html_report(type="text", section_header=f"Dataset Version: N/A", section_content_list=[
    f"Report run: {datetime.now()}"
])
include_in_html_report("header", section_content=f"Results", section_figure=2)

include_in_html_report("header", section_content=f"Comparison with other models", section_figure=2)


dff = pd.read_json('../../../results/results.json')

all_models_df = dff[dff.columns].T.sort_values("best score", ascending=False)

all_models_summary = all_models_df[['best score', 'best time', 'Mean Absolute Error Accuracy', 'Mean Squared Error Accuracy', 'R square Accuracy', 'Root Mean Squared Error', 'best run date', 'best method']]

include_in_html_report(type="dataframe", section_header="Comparison with all model performances", section_content=all_models_summary)


include_in_html_report("header", section_content=f"Appendix", section_figure=2)


df_summary_1_vs_2 = all_models_df[
    ['best score', 'best time', 'silver score', 'silver time', 'best method', 'silver method', 'best is shared']]
df_summary_1_vs_2

df_1_vs_2b = all_models_df[
    ['best score', 'best time', 'silver score', 'silver time', 'best params', 'silver params', 'best is shared','best method','silver method']]
df_1_vs_2b

d1 = all_models_df.iloc[0]['best params']
d2 = all_models_df.iloc[0]['silver params']
d99 = all_models_df.iloc[0]['_params']
d1['score'] = all_models_df.iloc[0]['best score']
d2['score'] = all_models_df.iloc[0]['silver score']
d99['score'] = all_models_df.iloc[0]['_score']
d1['time'] = all_models_df.iloc[0]['best time']
d2['time'] = all_models_df.iloc[0]['silver time']
d99['time'] = all_models_df.iloc[0]['_train time']
d1['date run'] = all_models_df.iloc[0]['best run date']
d2['date run'] = all_models_df.iloc[0]['silver run date']
d99['date run'] = all_models_df.iloc[0]['date']
d1['method'] = all_models_df.iloc[0]['best method']
d2['method'] = all_models_df.iloc[0]['silver method']
d99['method'] = all_models_df.iloc[0]['_method']

res = {}
for each in d1:
    res[each] = [d1[each]]
    if each in d2:
        if each not in res:
            d2[each]
            res[each]
        res[each].extend([d2[each]])
    else:
        res[each].append('not found')

including_most_recent = True
if including_most_recent == True:
    for each in d1:
        if each in d99:
            if each not in res:
                d2[each]
                res[each]
            res[each].extend([d99[each]])
        else:
            #d1[each] = d99[each]
            pass


run_comparison = pd.DataFrame.from_dict(res).T
if including_most_recent:
    run_comparison.columns = ['best model','runner-up model','most recent model']
else:
    run_comparison.columns = ['best model','runner-up model']
    
model = all_models_df.iloc[0].index


include_in_html_report(type="dataframe", section_header="Analyse something else", section_content=df_summary_1_vs_2)

include_in_html_report(type="dataframe", section_header="Analyse a 3rd thing", section_content=df_1_vs_2b)

include_in_html_report(type="dataframe", section_header=f"Analysing the hyperparameters for the best performing version (for {model})", section_content=run_comparison)

print('Nearly finished...')



print('Finished!')


# In[ ]:




