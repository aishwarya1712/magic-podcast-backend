#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install langchain


# In[12]:


import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

def encode_texts(texts, model, tokenizer):
    # Tokenize and encode the texts using the provided model and tokenizer
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # Forward pass to get embeddings
    with torch.no_grad():
        model_output = model(**encoded_texts)
    
    # Return the embeddings
    return model_output.last_hidden_state.mean(dim=1).numpy()

def create_faiss_index(text_vectors):
    # Convert text vectors to numpy array
    text_vectors_np = np.array(text_vectors).astype('float32')

    # Instantiate an index
    index = faiss.IndexFlatL2(text_vectors_np.shape[1])

    # Train the index
    index.add(text_vectors_np)

    return index

# Example usage:
# Replace 'distilbert-base-nli-stsb-mean-tokens' with the model of your choice
model_name = 'google/bert_uncased_L-2_H-128_A-2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# In[30]:


# Replace `your_texts` with the actual list of texts you want to index
your_texts = [
    """WASHINGTON (AP) — The Supreme Court should declare that Donald Trump is ineligible to be president again because he spearheaded the violent attack on the U.S. Capitol in an effort to overturn his 2020 election loss, lawyers leading the fight to keep him off the ballot told the justices on Friday.

In a filing filled with vivid descriptions of the Jan. 6, 2021, violence at the Capitol, the lawyers urged the justices not to flinch from doing their constitutional duty and to uphold a first-of-its-kind Colorado court decision to kick the 2024 Republican presidential front-runner off the state’s primary ballot.

“Nobody, not even a former President, is above the law,” the lawyers wrote.

The court will hear arguments in less than two weeks in a historic case that has the potential to disrupt the 2024 presidential election.

OTHER NEWS
Oregon State guard Talia von Oelhoffen (22) reacts during the first half of the teams' NCAA college basketball game against Colorado on Friday, Jan. 26, 2024, in Corvallis, Ore. (AP Photo/Amanda Loman)
Hunter and von Oelhoffen help No. 25 Oregon State defeat No. 3 Colorado 68-62
FILE - Rep. Lauren Boebert, R-Colo., a member of the House Freedom Caucus, speaks during a news conference, July 14, 2023, on Capitol Hill in Washington. Boebert's ex-husband, Jayson Boebert has been charged, Wednesday, Jan. 24, 2024, with assault and other charges in connection with two domestic altercations in early January. (AP Photo/Patrick Semansky, File)
Lauren Boebert’s ex-husband charged after 2 domestic incidents
Colorado guard J'Vonne Hadley (1) shoots from between Washington guard Koren Johnson (0) and Colorado center Eddie Lampkin Jr., second from right, during the second half of an NCAA college basketball game Wednesday, Jan. 24, 2024, in Seattle. Colorado won 98-81. (AP Photo/Lindsey Wasson)
Hadley posts career-high 24 points and Colorado smokes Washington 98-81 for 4th-straight win
The case presents the high court with its first look at a provision of the 14th Amendment barring some people who “engaged in insurrection” from holding public office. The amendment was adopted in 1868, following the Civil War.

In their plea to the court, the lawyers said, “Trump intentionally organized and incited a violent mob to attack the United States Capitol in a desperate effort to prevent the counting of electoral votes cast against him” after he lost the election to Democrat Joe Biden.

They called for a decision that makes clear that what happened on Jan. 6 was an insurrection, for which Trump bears responsibility. The president is covered by the constitutional provision at issue, and Congress doesn’t need to take action before states can apply it, the lawyers wrote.

The written filing includes extensive details of Trump’s actions leading up to Jan. 6, including his tweet on Dec. 19, 2020, in which he informed his followers of the planned protest on the day Congress would count the electoral votes and wrote, “Be there, will be wild.”

Then in his speech to supporters on Jan. 6, the lawyers wrote, “Trump lit the fuse.” The brief reproduces photographs of the mayhem from that day, including one of U.S. Capitol Police Officer Daniel Hodges pinned in a doorway during the attack.

Trump’s lawyers have argued that efforts to keep him off the ballot “threaten to disenfranchise tens of millions of Americans and ... promise to unleash chaos and bedlam” if other states follow Colorado’s lead.

The Colorado Supreme Court’s 4-3 ruling should be reversed for any of several reasons, Trump’s lawyers wrote, including that Trump did not engage in insurrection and that the presidency is not covered by the amendment. They also contend that Congress would have to enact legislation before states could invoke the provision to keep candidates off the ballot.

The justices are hearing arguments Feb. 8. Trump already has won the first two GOP presidential contests: the Iowa caucuses and the New Hampshire primary. Former U.N. Ambassador Nikki Haley is Trump’s sole remaining significant GOP opponent.

Still, both sides have said the court needs to act quickly so that voters know whether Trump is eligible to hold the presidency.

The court is dealing with the dispute under a compressed timeframe that could produce a decision before Super Tuesday on March 5, when the largest number of delegates in a day is up for grabs, including in Colorado.

A two-sentence provision in Section 3 of the 14th Amendment states that anyone who swore an oath to uphold the Constitution and then “engaged in insurrection” against it is no longer eligible for state or federal office. After Congress passed an amnesty for most of the former confederates the measure targeted in 1872, the provision fell into disuse until dozens of suits were filed to keep Trump off the ballot this year. Only the one in Colorado was successful.

Trump is separately appealing to state court a ruling by Maine’s Democratic secretary of state, Shenna Bellows, that he is ineligible to appear on that state’s ballot over his role in the Capitol attack. Both the Colorado Supreme Court and the Maine secretary of state’s rulings are on hold until the appeals play out.""",
    """RIFLE, Colo. (AP) — Republican U.S. Rep. Lauren Boebert ‘s ex-husband, Jayson Boebert, has been charged with assault and other counts in connection with two domestic altercations, including an argument with the congresswoman in a restaurant that spiraled out of control, and a fight with the former couple’s son, authorities said.

The altercations have drawn further attention to Lauren Boebert, a far-right politician with a combative political style.

After the argument at the restaurant on Jan. 6, Jayson Boebert was uncooperative with police and had to be forcibly removed from the business, according to an arrest affidavit. That led to the charges of disorderly conduct, third-degree criminal trespass and obstructing a peace officer, according to court documents.

The second incident, on Jan. 9, involved a physical fight with the couple’s 18-year-old son during which Jayson Boebert grabbed a rifle after the teen called authorities, according to a separate arrest affidavit. That led to charges of harassment, prohibited use of a weapon and third-degree assault, court documents show. Lauren Boebert was not present during the fight.

Jayson Boebert did not respond to calls and texts seeking comment Friday. He is scheduled for a court hearing next Monday.

Congresswoman Boebert is facing her own difficulties, including allegations during the first Republican primary debate Thursday night that she is “carpetbagging” by switching to Colorado’s 4th Congressional District from the 3rd District, where she’s lived and raised her family in the town of Silt.

Boebert argues that her candidacy in the old district threatened Republican control of the seat; she defeated the Democratic opponent, Adam Frisch, by only 546 votes in 2022. If Boebert had stayed in the 3rd District, she was expected to have a rematch in November with Frisch, who had already far out-fundraised her. """,
    """WASHINGTON (AP) — The Supreme Court should declare that Donald Trump is ineligible to be president again because he spearheaded the violent attack on the U.S. Capitol in an effort to overturn his 2020 election loss, lawyers leading the fight to keep him off the ballot told the justices on Friday.

In a filing filled with vivid descriptions of the Jan. 6, 2021, violence at the Capitol, the lawyers urged the justices not to flinch from doing their constitutional duty and to uphold a first-of-its-kind Colorado court decision to kick the 2024 Republican presidential front-runner off the state’s primary ballot.

“Nobody, not even a former President, is above the law,” the lawyers wrote.

The court will hear arguments in less than two weeks in a historic case that has the potential to disrupt the 2024 presidential election.

OTHER NEWS
Oregon State guard Talia von Oelhoffen (22) reacts during the first half of the teams' NCAA college basketball game against Colorado on Friday, Jan. 26, 2024, in Corvallis, Ore. (AP Photo/Amanda Loman)
Hunter and von Oelhoffen help No. 25 Oregon State defeat No. 3 Colorado 68-62
FILE - Rep. Lauren Boebert, R-Colo., a member of the House Freedom Caucus, speaks during a news conference, July 14, 2023, on Capitol Hill in Washington. Boebert's ex-husband, Jayson Boebert has been charged, Wednesday, Jan. 24, 2024, with assault and other charges in connection with two domestic altercations in early January. (AP Photo/Patrick Semansky, File)
Lauren Boebert’s ex-husband charged after 2 domestic incidents
Colorado guard J'Vonne Hadley (1) shoots from between Washington guard Koren Johnson (0) and Colorado center Eddie Lampkin Jr., second from right, during the second half of an NCAA college basketball game Wednesday, Jan. 24, 2024, in Seattle. Colorado won 98-81. (AP Photo/Lindsey Wasson)
Hadley posts career-high 24 points and Colorado smokes Washington 98-81 for 4th-straight win
The case presents the high court with its first look at a provision of the 14th Amendment barring some people who “engaged in insurrection” from holding public office. The amendment was adopted in 1868, following the Civil War.

In their plea to the court, the lawyers said, “Trump intentionally organized and incited a violent mob to attack the United States Capitol in a desperate effort to prevent the counting of electoral votes cast against him” after he lost the election to Democrat Joe Biden.

They called for a decision that makes clear that what happened on Jan. 6 was an insurrection, for which Trump bears responsibility. The president is covered by the constitutional provision at issue, and Congress doesn’t need to take action before states can apply it, the lawyers wrote.

The written filing includes extensive details of Trump’s actions leading up to Jan. 6, including his tweet on Dec. 19, 2020, in which he informed his followers of the planned protest on the day Congress would count the electoral votes and wrote, “Be there, will be wild.”

Then in his speech to supporters on Jan. 6, the lawyers wrote, “Trump lit the fuse.” The brief reproduces photographs of the mayhem from that day, including one of U.S. Capitol Police Officer Daniel Hodges pinned in a doorway during the attack.

Trump’s lawyers have argued that efforts to keep him off the ballot “threaten to disenfranchise tens of millions of Americans and ... promise to unleash chaos and bedlam” if other states follow Colorado’s lead.

The Colorado Supreme Court’s 4-3 ruling should be reversed for any of several reasons, Trump’s lawyers wrote, including that Trump did not engage in insurrection and that the presidency is not covered by the amendment. They also contend that Congress would have to enact legislation before states could invoke the provision to keep candidates off the ballot.

The justices are hearing arguments Feb. 8. Trump already has won the first two GOP presidential contests: the Iowa caucuses and the New Hampshire primary. Former U.N. Ambassador Nikki Haley is Trump’s sole remaining significant GOP opponent.

Still, both sides have said the court needs to act quickly so that voters know whether Trump is eligible to hold the presidency.

The court is dealing with the dispute under a compressed timeframe that could produce a decision before Super Tuesday on March 5, when the largest number of delegates in a day is up for grabs, including in Colorado.

A two-sentence provision in Section 3 of the 14th Amendment states that anyone who swore an oath to uphold the Constitution and then “engaged in insurrection” against it is no longer eligible for state or federal office. After Congress passed an amnesty for most of the former confederates the measure targeted in 1872, the provision fell into disuse until dozens of suits were filed to keep Trump off the ballot this year. Only the one in Colorado was successful.

Trump is separately appealing to state court a ruling by Maine’s Democratic secretary of state, Shenna Bellows, that he is ineligible to appear on that state’s ballot over his role in the Capitol attack. Both the Colorado Supreme Court and the Maine secretary of state’s rulings are on hold until the appeals play out.""","""COLUMBIA, S.C. (AP) — The Republican National Committee has pulled a resolution to consider declaring Donald Trump the party’s “presumptive 2024 nominee” before he formally clinches the requisite number of delegates, a person familiar with the decision said Thursday.

News of the withdrawal came shortly after Trump posted on his Truth Social site that, while he “greatly” appreciated the notion, he felt, “for the sake of PARTY UNITY, that they should NOT go forward with this plan, but that I should do it the ‘Old Fashioned’ way, and finish the process off AT THE BALLOT BOX.”

The measure, according to a draft obtained Thursday by The Associated Press, had said it “declares President Trump as our presumptive 2024 nominee for the office of President of the United States and from this moment forward moves into full general election mode welcoming supporters of all candidates as valued members of Team Trump 2024.”

The withdrawal was confirmed by a person familiar with the decision who was not authorized to publicly discuss the proposal and spoke on condition of anonymity Thursday night.

OTHER NEWS
FILE - Republican presidential candidate former President Donald Trump speaks at a primary election night party in Nashua, N.H., Jan. 23, 2024. A jury is trying to determine if the $83 million defamation judgement will be enough to deter Trump from continuing his attacks on E. Jean Carroll. The jury hit Trump with the exorbitant defamation award Friday, Jan. 26, finding the former president maliciously damaged Carroll's reputation after she accused him of sexually assaulting her in a Manhattan department store. (AP Photo/Matt Rourke, File)
Donald Trump is on the hook for $88.3 million in defamation damages. What happens next?
The U.S. Supreme Court is photographed on Wednesday, Jan. 17, 2024, in Washington. (AP Photo/Mariam Zuhaib)
Supreme Court is urged to rule Trump is ineligible to be president again because of the Jan. 6 riot
FILE - Fulton County District Attorney Fani Willis speaks during an Associated Press interview on Tuesday, Dec. 12, 2023, in Atlanta. The Georgia state Senate voted on Friday, Jan. 26, 2024, to create a special committee to investigate Willis, with Republicans claiming she has personally benefitted from improperly conveying public money to special prosecutor Nathan Wade. (AP Photo/Brynn Anderson, File)
Georgia Senate passes a panel with subpoena power to investigate District Attorney Fani Willis
If approved, the measure would have further solidified Trump’s control of the party and its operation at a time when former United Nations Ambassador Nikki Haley is still competing against Trump for the GOP nomination.

RNC Chair Ronna McDaniel had earlier signaled her approval for the resolution. On Tuesday, after Haley finished second to Trump in New Hampshire, McDaniel said that while she felt the former ambassador had “run a great campaign,” Republicans “need to unite around our eventual nominee, which is going to be Donald Trump.”

The resolution had been expected to be discussed at the RNC’s winter meeting in Las Vegas next week, even though only two states have voted and the former president had nowhere near the requisite number of delegates to secure the nomination.

Haley’s camp said Thursday that it wasn’t up to the RNC to decide who the GOP nominee would be.

“Who cares what the RNC says? We’ll let millions of Republican voters across the country decide who should be our party’s nominee, not a bunch of Washington insiders,” said campaign spokesperson Olivia Perez-Cubas.

The AP has a policy to not refer to any candidate as the “presumptive nominee” until he or she has captured the number of delegates needed to win a majority vote at the national party conventions this summer. The earliest that could happen is March.

But there were no party rules prohibiting the RNC from making such a move. If it had been adopted, it could have given the Republican Party a jump-start on planning a general election matchup with Democratic President Joe Biden, who has begun framing his reelection campaign as a 2020 rematch against Trump.

Republican presidential candidate former UN Ambassador Nikki Haley speaks during a campaign event at The North Charleston Coliseum, Wednesday, Jan. 24, 2024, in North Charleston, S.C. (AP Photo/Sean Rayford)
Republican presidential candidate former UN Ambassador Nikki Haley speaks during a campaign event at The North Charleston Coliseum, Wednesday, Jan. 24, 2024, in North Charleston, S.C. (AP Photo/Sean Rayford)

There was also precedent for the committee to declare a candidate the presumptive nominee before winning the 1,215 requisite delegates to clinch the nomination. Then-RNC Chair Reince Priebus did so with Trump in May 2016.

Despite losing both the Iowa and New Hampshire contests to Trump, Haley has argued that her performance — outlasting all the other Trump rivals — shows the strength of her candidacy.

Trump currently has 32 delegates to Haley’s 17. There is one delegate left to be assigned after the New Hampshire contest.

During a rally Wednesday night in her home state of South Carolina, Haley — the former governor — noted that her campaign had brought in more than $1 million since her second-place finish in New Hampshire. Trump followed up with a remark that appeared aimed at intimidating her donors.

“Anybody that makes a ‘Contribution’ to Birdbrain, from this moment forth, will be permanently barred from the MAGA camp,” Trump wrote, using the nickname he has crafted for Haley and the abbreviation for his “Make America Great Again” slogan. “We don’t want them, and will not accept them, because we Put America First, and ALWAYS WILL!”

Haley’s campaign said Thursday that it raised an additional $1.2 million “after Trump’s unhinged pledge to ‘permanently bar’ any individual who contributed to Haley’s campaign.”

“Donald Trump’s threats highlight the stark choice in this election: personal vendettas or real conservative leadership,” said Haley spokesperson AnnMarie Graham-Barnes. “Trump’s scheme blew up in his face. The contributions to the Haley campaign are pouring in — proof that people are sick of the drama and are rallying behind Nikki’s vision for a strong and proud America.”

Trump’s dismissal of any Haley donors had no effect on T.J. Petrizzo, a former top Capitol Hill staffer and now lobbyist who supports Haley.

“That’s something out of a ‘Godfather’ movie. Never betray the family? Come on,” he added. “You’ve got to play this through.”

Petrizzo said he understands that some Republicans may be ready to pivot to a head-to-head contest between Trump and Biden, but he notes that there is a lot of time left before a general election.

“I’ve heard a lot of elected officials in the Republican Party, including the RNC chair, say, ‘We need to rally around a candidate.’ That this is going to be our candidate. ‘It was chosen by Iowa and New Hampshire, so we must go ahead and rally around Trump,’” Petrizzo said. “Well, there’s 285 days until the election. There’s plenty of time on the clock.”"""
]


# In[ ]:


# np.delete(your_texts,2)


# In[49]:


your_texts = your_texts[:2] + your_texts[3:]
print(your_texts)  # Output: [10, 20, 40]


# In[50]:


# Encode texts and create Faiss index
text_vectors = encode_texts(your_texts, model, tokenizer)
faiss_index = create_faiss_index(text_vectors)
faiss.write_index(faiss_index, "my_faiss_index")


# In[51]:


loaded_faiss_index = faiss.read_index("my_faiss_index")
query_text = "what's happening with trump"


# In[52]:


# Replace 'your_query_text' with the actual query text
query_text = "what's happening with trump"

# Encode the query text using the same model and tokenizer
query_vector = encode_texts([query_text], model, tokenizer)

# Perform a similarity search
k = 3  # Number of nearest neighbors to retrieve
distances, indices = loaded_faiss_index.search(query_vector, k)

# Print the results
print("Nearest neighbors:")
for i, index in enumerate(indices[0]):
    print(f"Rank {i+1}: Index {index}, Distance: {distances[0][i]}")


# In[53]:


final_news_array = []
for i in range(k):
    final_news_array.append(your_texts[indices[0][i]])


# In[54]:


np.save("my_array.npy", final_news_array)


# In[ ]:




