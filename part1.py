import pandas as pd
import bert.helpers as h

pd.options.display.max_colwidth = 25000

tr = h.train_data()
te = h.test_data()
st = h.stance_data()

tr.head()

op_about_trg = "1.  The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target."
no_op_about_trg = "2. The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target."
no_op = "3.  The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)"

tr[(tr.Target == "Hillary Clinton") & (tr.Sentiment == "other")].Stance.value_counts()

te.columns
te[(te.Target == "Feminist Movement") & (te.Sentiment == "other")].Stance.value_counts()

tr.Target.value_counts()

tr.Stance.value_counts()

tr[(tr.Target == "Hillary Clinton") & (tr["Opinion Towards"] == no_op)].head(30)

tr.Stance.value_counts()
tr[(tr.Target == "Hillary Clinton") & (tr.Stance == "AGAINST") & (tr.Sentiment == "pos")].shape

tr["Opinion Towards"].value_counts() # 2740
tr.Stance.value_counts()

tr.head()
tr.Target.value_counts()
tr.shape

te.head()
te.Target.value_counts()

st["controversial trending issue"].value_counts()
st["stance"] = "NONE"
st.sample(50).to_csv(h.dataset_path + "stance_predict.csv")

# test_set = pd.concat([
#     st[st["controversial trending issue"] == topic].sample(10)
#     for topic in h.TOPICS
#     if topic is not None
# ])
# test_set.to_csv(h.dataset_path + "stance_test_random.csv")
