from flask import Flask, request, jsonify, send_file
from flask_cors import CORS,cross_origin
import torch
import openai
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import elevenlabs


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
# news_summary = ""
# news_title = ""
app.config["NEWS_SUMMARY"] = ""
app.config["TITLE"] = ""

@app.route('/api/data')
def get_data():
    # This is where you would typically query your database
    data = {"message": "Hello from Flask Backend 2!"}
    return jsonify(data)


@app.route('/x1', methods=['POST'])
@cross_origin()
def handle_post():
    # Get JSON data from the request
    data = request.json

    # Data from UI
    topics = data['topics']
    length = data['length']
    
    result = "You sent: " + str(topics) + " " + str(length)

    #IndexCode
    openai.api_key  = "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"

    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

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
    model_name = 'google/bert_uncased_L-2_H-128_A-2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Replace 'your_query_text' with the actual query text
    query_text = topics[0]

    # Encode the query text using the same model and tokenizer
    query_vector = encode_texts([query_text], model, tokenizer)

    # Perform a similarity search
    my_faiss_index = faiss.read_index("faiss_index_12news")
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = my_faiss_index.search(query_vector, k)
    
    arr = ['''California restaurant incorporates kitchen robots and AI

It's long since been the vision of Hollywood that robots and humans would be side-by-side. In some Hollywood films, this partnership is a recipe for catastrophe, but in this suburb not far from Tinseltown, AI and robots are working in a restaurant kitchen. 

The restaurant, called CaliExpress, can be found in the heart of Pasadena, California. It's the first time that so much technology has been in one place, according to business owner Vic Aulakh. 

The burger joint uses a grill robot by Cucina and "Flippy," a robot from Miso Robotics that can handle frying. Both robots can make quite a lot of food: Flippy can make 250 pounds of French fries an hour, and the grill robot can cook about 100 patties in that time. Neither machine needs a break or a day off. 

"We can't get enough people to come out and work on the fryer and grills," Aulakh said. "They're dangerous jobs and this automation helps solve a lot of those issues we're having." 

Flippy can also be found in chain restaurants like White Castle and Jack in the Box. Rob Anderson, a co-founder of Miso Robotics, said that the company started working on the automaton six years ago. 

"Flippy is really good at repetitive tasks like operating the fryer in those dangerous environments," Anderson said. "That way, the people working the restaurant can focus on the human element." 

CaliExpress isn't just using robot chefs. The ordering system at the restaurant is powered by artificial intelligence and uses facial recognition software run by Pop ID to keep track of food choices and payment. The technology's creator says that the system is not used for surveillance. 

There will be some human employees at CaliExpress, too. The restaurant plans on only hiring two "back of house" employees to put finishing touches on a meal. That's a fraction of hiring for a non-automated kitchen. In a state where the minimum wage for fast food workers will soon be $20 an hour, having less people to pay can be a bonus for businesses. 

"It does reduce some jobs, but there's so many more jobs created with the technology as well as maintaining this technology," Aulakh said. 

A study published this week by researchers at MIT found that just a small amount of jobs could be done by AI, but in most cases, it doesn't make economic sense to use AI to do the necessary work. 


However, a December 2023 government report listed AI as a threat to financial stability. One study found that 82 percent of restaurant jobs could be handled by robotics, an important statistic in an industry that often struggles with understaffing. 

Both robots are rented, and technicians monitor them 24/7. Those technicians can take over cooking if something goes wrong with the robots during meal times. 

"If for any reason Flippy's not working, they can flip the barrier up and cook manually as they would normally," Anderson said. 

Aulakh said that hopefully, the robots will also affect the taste of the restaurant's food - for the better. 

"Now, each burger is being cooked precisely the same way every time," Aulakh said.
''', '''General purpose humanoid robots? Bill Gates is a believer
Brian Heater@bheater / 2:06 PM PST•January 25, 2024
 Comment
Bill Gates on stage at TC Sessions: Climate 2022
Image Credits: TechCrunch
The robotics industry loves a good, healthy debate. Of late, one of the most intense ones centers around humanoid robots. It’s been a big topic for decades, of course, but the recent proliferation of startups like 1X and Figure — along with projects from more established companies like Tesla — have put humanoids back in the spotlight.

Proponents of the form factor point out that we’ve built our world to fit ourselves, so it makes sense that we would build robots like ourselves to fit it. There are also certain advantages in terms of reach, the ability to walk up stairs and the dexterity that comes with our design.

Of course, anyone who tells you that the human body is the pinnacle of organic machinery is either misinformed or lying to you. I’ve spent the past year dealing with degenerative disc disease, a perfect testament to our imperfect design.

Agility Robotics picking and stacking
Image Credits: Agility Robotics(opens in a new window)


The form factor also flies in the face of decades of conventional wisdom that has championed single-purpose robots – that is, machines that are built to do one thing extremely well a whole lot of times. And then there’s the whole “general purpose” part, which tends to get tossed around with little thought to its underlying complexity.

Humanoid robots can, however, now claim a big tech name among their ranks. Bill Gates this week issued a list of “cutting-edge robotics startups and labs that I’m excited about.” Among the names are three companies focused on developing humanoids. The first and most prominent is Agility, whose Digit robot looks the least like a human of the three. Also included are Apollo-maker Apptronik and UCLA’s RoMeLa (Robotics and Mechanisms Lab), which is behind the soccer-playing ARTEMIS.

Here’s what Gates has to say about Apptronik,


What’s more useful: multiple robots that can each do one task over and over, or one robot that can do multiple tasks and learn to do even more? To Apptronik, an Austin-based start-up that spun out of the human-centered robotics lab at the University of Texas, the answer is obvious. So they’re building “general-purpose” humanoid bi-pedal robots like Apollo, which can be programmed to do a wide array of tasks—from carrying boxes in a factory to helping out with household chores.

Astra
Image Credits: Apptronik(opens in a new window)


Writing about Agility, he notes, “If we want robots to operate in our environments as seamlessly as possible, perhaps those robots should be modeled after people.” Digit is currently ahead of the pack in terms of real-world deployments, including a recent pilot at Amazon warehouses that helped set the stage for Figure’s recent BMW deal.

Other firms mentioned in the piece include robotic perception firm Field AI and Tevel, which builds apple-picking drones.

An endorsement like this might not move the needle too far in the humanoid direction, and Gates is very much not a roboticist. It is, however, enlightening to see the form factor continue to gain more mainstream legitimacy by the day.''', '''A robot surgeon is headed to the ISS to dissect simulated astronaut tissue
News
By Monisha Ravisetti published 1 day ago
It'll be joined by a bunch of equally impressive experiments such as machine learning devices and artificial retina prototypes.

 
 
 
 
 
 Comments (0)
a robotic hand-like device reaching towards rubber bands in a lab
MIRA, Virtual Incision’s miniaturized robotic assisted surgery system, is pictured in position to reach rubber bands serving as simulated surgical tissue. (Image credit: NASA/Virtual Incision)
Very soon, a robot surgeon may begin its orbit around our planet — and though it won't quite be a metallic, humanoid machine wearing a white coat and holding a scalpel, its mission is fascinating nonetheless.

On Tuesday (Jan. 30), scientists will be sending a slew of innovative experiments to the International Space Station via Northrop Grumman's Cygnus spacecraft. It's scheduled to launch no earlier than 12:07 p.m. ET (1707 GMT) and, if all goes to plan, arrive at the ISS a few days later on Feb. 1.

Indeed one of the experiments onboard is a two-pound (0.9-kilogram) robotic device, about as long as your forearm, with two controllable arms that respectively hold a grasper and a pair of scissors. Developed by a company named Virtual Incision, this doctor robot of sorts is built to someday be able to communicate with human doctors on the ground while inserting itself into an astronaut patient to conduct medical procedures with high accuracy. 

"The more advanced part of our experiment will control the device from here in Lincoln, Nebraska, and dissect simulated surgical tissue on orbit," Shane Farritor, co-founder of Virtual Incision, said during a presentation about Cygnus on Friday.

For now, as it's in preliminary stages, it's going to be tested on rubber bands — but the team has high hopes for the future as missions to the moon, Mars and beyond start rolling down the space exploration pipeline. Remote space medicine has become a hot topic during the last few years as space agencies and private space companies lay plans for a variety of future crewed space missions. 

Related: International Space Station will host a surgical robot in 2024

NASA's Artemis Program, for instance, hopes to have boots on the moon in 2026 — plus, that's supposed to pave the way for a day on which humanity can say they've reached the Red Planet. And together, those missions are expected to pave the way for a far future in which humanity embarks on deeper space travel, perhaps to Venus or, if we're really dreaming, beyond the solar system. So to make sure astronauts remain safe in space — an environment they're literally not made to survive in — scientists want to make sure space-based medical treatment sees advancement in tandem with the rockets that'll take those astronauts wherever they're going. 

A quick example that comes to mind is how, in 2021, NASA flight surgeon Josef Schmid was "holoported" to the ISS via HoloLens technology. It's sort of like virtual reality meets FaceTime meets augmented reality, if that makes sense.

Hologram medical professionals on the ISS on Oct. 8, 2021. From the left, Andrew Madrid, Dr. Fernando De La Pena Llaca, Rlhab Sadik, Dr. Joe Schmid, Kevin Bryant, Mackenzie Hoffman and Wes Tarkington.

Hologram medical professionals seen on the ISS on Oct. 8, 2021, through HoloLens technology. (Image credit: ESA (European Space Agency) astronaut Thomas Pesquet)
However, as the team explains, not only could this robotic surgery mission benefit people exploring the void of space, but also those living right here on Earth. "If you have a specialist who's a very good surgeon, that specialist could dial into different locations and help with telesurgery or remote surgery," Farritor said. "Only about 10% of operating rooms today are robotic, but we don't see any reason that shouldn't be 100%."

a silver box with knobs on it and wires extending from it with buttons reading "autonomous," "power" and "ethernet"

The investigation locker for the Robotic Surgery Tech Demo investigation is shown on the ground. The locker was designed so that the astronauts can initiate "autonomous mode" and switch to "telesurgery mode." The miniaturized robotic surgeon is housed inside the microwave-sized locker and will be controlled via remote human control and pre-programmed movements.  (Image credit: NASA/Virtual Incision)
This would be a particularly crucial advantage for hospitals in rural areas where fewer specialists are available, and where operating rooms are limited. In fact, as Farritor explained, not only is Virtual Incision funded by NASA but also by the military. "Both groups want to do surgery in crazy places," he said, "and our small robots kind of lend themselves to mobility like that."

a human hand holding a robotic device resembling a claw or hand

The robotic surgery device is shown outside of its investigation locker. The Robotic Surgery Tech Demo will test the performance of a small robot that can be remotely controlled from Earth to perform surgical procedures. Researchers plan to compare procedures in microgravity and on Earth to evaluate the effects of microgravity and time delays between space and ground. (Image credit: NASA/Virtual Incision)
What else is going up? 
The little robot doctor will be far from alone on the Cygnus spacecraft as it heads to the ISS; during the same presentation in which Farritor discussed Virtual Incision, other experts talked about what they'll be sending up come Monday.

For one, it'll have a robot friend joining it in the orbital laboratory — a robotic arm. This arm has already been tested within the station's constraints before, but with this new mission the team hopes to test it in fully unpressurized conditions. 

"Unplugging, replugging, moving objects, that's the kind of stuff that we did with the first investigation," said May Murphy, the director of programs at company NanoRacks. "We're kind of stepping up the complexity ... we're going to switch off which tools we're using, we'll be able to use screwdriver analogs and things like that; that will enable us to do even more work." 

"We can look at even beyond just taking away something that the crew would have to spend time working on," she continued. "Now, we also have the capacity to do additional work in harsher environments we don't necessarily want to expose the crew to." 

The European Space Agency, meanwhile, will be sending a 3D-printer that can create small metal parts. The goal here is to see how the structure of 3D-printed metal fares in space when compared to Earth-based 3D-printed metal. 3D-printed semiconductors, key components of most electronic devices, will be tested as well for a similar reason.

"When we talk about having vehicles in space for longer periods of time without being able to bring supplies up and down, we need to be able to print some of these smaller parts in space, to help the integrity of the vehicle over time," said Meghan Everett, NASA's ISS program deputy scientist. 

Per Everett, this could also help scientists learn whether some sorts of materials that aren't 3D-printable on Earth can be 3D-printed in space. "Some preliminary data suggests that we can actually produce better products in space compared to Earth which would directly translate to better electronics in energy producing capabilities," she said.

Another experiment getting launched on Monday looks at the effects of microgravity on bone loss. Known as MABL-A, it will look at the role of what're known as mesenchymal cells (associated with bone marrow) and how that might change when exposed to the space environment. This could offer insight into astronaut bone loss — a well-documented, major issue for space explorers — as well as into the dynamics of human aging. "We will also look at the genes that are involved in bone formation and how gravity affected them," said Abba Zubair, a professor of Laboratory Medicine and Pathology at Mayo Clinic.

Lisa Carnell, division director for NASA's Biological and Physical Sciences Division, spoke about the Apex-10 mission headed up, which will see how plant microbes interact in space. This could help decode how to increase plant productivity on Earth, too. 

Computers and retinas
Two of the other key experiments discussed during the presentation include a space computer and an artificial eye — well, an artificial retina, to be exact. We'll start with the latter.

Nicole Wagner, CEO of a company named LambdaVision, has a staggering goal: To restore vision to the millions of patients that are blinded by end stage retinal degenerative diseases like macular degeneration and retinitis pigmentosa. 

To do this, she and her team are trying to develop a protein-based artificial retina that's built through a process known as "electrostatic layer-by-layer deposition." In short, this consists of depositing multiple layers of a special kind of protein onto a scaffold. "Think of the scaffold almost like a tightly woven piece of gauze," Wagner said. 

However, as she explains, this process on Earth can be impeded by the effects of gravity. And any imperfections in the layers can pretty much ruin the artificial retina's performance. So …  what about in microgravity? To date, LambdaVision has flown more than eight missions to the ISS, she says, and the experiments have shown that microgravity does indeed generate more homogenous layers and therefore better thin films for the retina.

"In this mission," she said, "we're looking at sending a powdered form of bacteriorhodopsin to the ISS that will then be resuspended into a solution, and we will be using special instruments, in this case spectrometers, to look at the protein quality and purity on the International Space Station, as well as to validate this process used to get the protein into solution."

Could you imagine if doctors would be able to commission a few artificial retinas to be developed in space someday, then delivered to the ground for implantation into a patient. And that this whole process could give someone their sight back?

As for the space computer, Mark Fernandez, principal investigator for the Spaceborne Computer-2 project, posed a hypothetical. "Astronauts go on a spacewalk, and after their work day, the gloves are examined for wear-and-tear,' he said. "This must be done by every astronaut, after every spacewalk, before the gloves can be used again."

Normally, Fernandez explains, the team takes a bunch of high-resolution photographs of the potentially contaminated gloves, then sends those images out for analysis. 

The Cygnus NG-19 Laurel Clark space freighter captured by the space station's robotic arm.

(Image credit: NASA)
RELATED STORIES:
— 12 space medicine findings from 2023 that could help astronauts reach Mars one day

— Medical emergencies will happen in deep space. Here's how Canada is getting ready

— 'There's nothing magical that happens in 2030': NASA eyes possible ISS extension for astronaut missions

This analysis, he says, typically takes something like five days to finish and return. So, hoping to solve the problem, the team developed an AI model in collaboration with NASA and Microsoft that can do the analysis straight on the station and flag areas of concern. Each takes about 45 seconds to complete. "We're gonna go on from five days to just a few minutes," he said, adding that the team also did DNA analysis typically conducted on the space station in about 12 minutes. Normally, he emphasized, that'd take months.

But, the team wants to make sure Spaceborne Computer-2's servers will function properly while on the ISS, hence the Cygnus payload. This will mark the company's third ISS mission.

"The ISS National Lab has so many benefits that it's attributing to our nation," Carnell said. "It creates a universe of new possibilities for the next generation of scientists and engineers." ''', '''Western US Seeing Extreme Weather 'Unprecedented' in 500 Years
Jan 24, 2024 at 2:00 PM EST
AD Loading ...

Video Player is loading.
By Robyn White
Nature Reporter
FOLLOW
4
Droughts in the Western U.S. have risen to an "unprecedented" level since the 16th century, a new study has revealed.

"Hot drought" refers to concurrent drought conditions that are stretching across the country, largely because of climate change. The study from multi-institutional researchers, published in the journal Science, found that these severe conditions have been increasing in not just severity but how often they have occurred over the past 100 years.

To get these findings, lead author Karen King and colleagues combined summer temperatures in the U.S. from 1553 to 2020 by analyzing tree rings, which show temperature changes through soil moisture. They discovered that the past 20 years in the Western U.S. have been the hottest in over five centuries.

"If anthropogenic warming continues according to model projections, we might assume that anthropogenic drying has only just begun," King, an assistant professor of physical geography at the University of Tennessee, told Newsweek. "Furthermore, we might expect the frequency of compound hot and dry conditions—hot drought—to increase over the course of this century."

Drought conditions 
A stock photo pictures ground cracked with drought. New research shows that the past 20 years in the Western U.S., which has been severely afflicted by drought, have been the hottest in over five centuries.
PIYASET/GETTY
These hot temperatures have increased the soil moisture, which in turn has contributed to the severity and frequency of drought. The researchers have linked the notable increase in drought to human activities.

Sign up for Newsletter
NEWSLETTER
The Bulletin
Your daily briefing of everything you need to know
Email address
 
By clicking on SIGN ME UP, you agree to Newsweek's Terms of Use & Privacy Policy. You may unsubscribe at any time.
"Individual climatic events can have serious effects on agriculture, infrastructure, and ecosystems, but compounding hazards can result in cascading and intensified consequences for these systems," the authors write in the study.

They went on: "The combination of anomalous heat with rainfall deficits have already led to droughts that are substantially more intense because of high temperatures and elevated vapor pressure deficits and have thus been called hot droughts."

The research is easy to believe, considering the severe effects of drought we are already seeing in the Western U.S. The region is facing a severe water crisis as megadrought continues to plague the area.

The Colorado River Basin is seeing a concerning drop in water levels as drought causes a lack of seasonal precipitation. Important Colorado River reservoirs like Lake Mead are showing how dire the situation is. In the summer of 2022, the reservoir reached its lowest level ever recorded. Since then, the water levels have remained at only around 30 percent capacity.

READ MORE
Rhinos at risk as temperatures set to become deadly
Lake Mead set for big drop in water levels in 2024
NASA map reveals freezing temperatures sweeping across the US
"The increasing prevalence of hot drought over the 20th and 21st centuries has important implications for future regional climate change adaptation strategies and for water resource management, particularly in the most historically drought-prone regions," the study said.

"This study aims to evaluate the relationship between summer average maximum temperatures and summer soil moisture over the last 500 years," King said. "Our main research questions are, [firstly] how might we characterize or quantify the relationship between summer maximum temperature and summer soil moisture during this modern period of megadrought conditions for much of western North America compared to during past megadrought periods over the last 500 years? [And secondly,] has the frequency of compound hot and dry summers increased over the 20th-21st centuries compared to the past ca. 500 years?"

Experts are already aware that drought is wreaking havoc across the globe. In December 2023, the United Nations reported that drought is starting to become an unprecedented planetary-scale emergency.

At the time of this research, Daniel Tsegai, program officer at the global policy advocacy and regional cooperation unit at the United Nations Convention to Combat Desertification, told Newsweek that drought is hitting many areas of the U.S.

"Our research team is currently working to expand the Western North American Temperatures Atlas to provide complete spatiotemporal coverage of past summer temperature estimates for the North American continent, spanning at least the last 500 years," King said. "Completion of the full North American Temperature Atlas will allow for broader spatial comparisons of compound climate extremes though time."

Update 01/25/24, 5:59 a.m. ET: This article was updated with direct quotes from Karen King.''', '''The Daily Weather Update from FOX Weather: South faces more flooding; Northeast faces messy mix
Start your day with the latest weather news – Flooding rain won’t let up in the South on Friday. Meanwhile, the Northeast prepares for a sloppy storm that will bring not just rain but also a wintry mix.
By Aaron BarkerSource FOX Weather
Facebook
Twitter
Email
Copy Link
Waves of heavy rain, storms continue Friday along Gulf Coast
Days of rain have left areas of the Deep South drenched and we're looking at even more rain Friday. Showers and thunderstorms will pop up again today from Louisiana to Mississippi to Alabama and Georgia.

Welcome to the Daily Weather Update from FOX Weather. It’s Friday, Jan. 26, 2024. Start your day with everything you need to know about today's weather. You can also get a quick briefing of national, regional and local weather whenever you like with the FOX Weather Update podcast.

Flooding threat remains for South
Days of heavy rain across the South has led to widespread flooding, which has damaged roads and led to at least two deaths and several water rescues. The rain continues Friday, with the heaviest expected to stretch from Louisiana to Georgia. The stretch of soggy weather should come to an end this weekend, but not before severe weather marches across parts of the Southeast.


The excessive rainfall outlook for Friday.
(FOX Weather)

 
Sloppy storm to bring messy mix of weather to Northeast
Parts of the Northeast and New England are facing a weekend of weather that will feature a little bit of everything. Winter weather is possible for much of the region, but the highest accumulations of snow are expected in Massachusetts and New Hampshire by Monday. Otherwise, most places can expect rain. It could end as snow for New York City, but only minor accumulations are expected.


The snowfall forecast for the Northeast this weekend.
(FOX Weather)

 
Atmospheric river storms target West Coast again
Another series of atmospheric rivers is set to bring nasty weather to the West Coast next week. Flooding rain, heavy snow, high winds and pounding surf are all possible during the storms. The first arrives in the Northwest on Friday. California looks to get hit hard by the final one that starts Tuesday.''', '''Living on the edge (literally!): Homeowners with stunning ocean views forced to flee before crumbling cliffs battered by extreme weather dump properties into the water
Oceanfront homes are at risk of ending up in the water due to increased incidents of extreme weather
Images shows homes in Rhode Island, California and Maine all on the brink of collapse following a particularly violent winter
Do YOU live on a cliffside property? Get in touch: money@dailymail.com 
By HELENA KELLY ASSISTANT CONSUMER EDITOR FOR DAILYMAIL.COM

UPDATED: 18:39 EST, 27 January 2024

e-mail
38
shares
556

View comments  e-mail
Top
Home
556

View comments
Many Americans surely dream of owning a stunning oceanfront property with panoramic views of the California coastline. 

But for the residents of Pacifica, San Mateo, that dream is becoming a nightmare as extreme storms decimate the cliffs these million-dollar properties sit on - leaving them at risk of crumbling into the ocean. 

Now, owners are being forced to move.

It is a problem facing homeowners in coastal areas all across the US. Large swathes of the country have been hit with record flooding and tornadic storms in recent weeks, with images of homes submerged in water plastered across the news. 

Those by the sea, however, face a longer-term and more sinister threat of coastal erosion which is already ripping apart the very foundation their homes are built on. 

It means many properties are uninsurable while facing the very real prospect they could soon be underwater. Now, owners are looking to move.

People stand along the shore to watch massive waves as high surf and coastal flood warning have been issued, in Pacifica California. Residents in the area complain they are being forced out by coastal erosion
+15
View gallery
People stand along the shore to watch massive waves as high surf and coastal flood warning have been issued, in Pacifica California. Residents in the area complain they are being forced out by coastal erosion

Extreme storms this winter have decimated the cliffs many coastal homes sit on- meaning they are at risk of crumbling into the ocean in just a matter of years. Pictured: Pacifica, California in January 2024
+15
View gallery
Extreme storms this winter have decimated the cliffs many coastal homes sit on- meaning they are at risk of crumbling into the ocean in just a matter of years. Pictured: Pacifica, California in January 2024

Footage shows homes in Narragansett, Rhode Island, on the brink of collapse following two back-to-back storms
+15
View gallery
Footage shows homes in Narragansett, Rhode Island, on the brink of collapse following two back-to-back storms

TRENDING

Trump plans to challenge 'insane' issue in $83.3 MILLION jury verdict
5.8k viewing now

Music star and King's anti-violence ambassador assaulted band member
6.8k viewing now

Tennis legend is completely unrecognizable as he is spotted in LA
5.3k viewing now

Pacifica resident Joan Levin says the issue has forced her to move out of her home of 45 years. Properties in the area are currently worth $1,129,444, according to data from Redfin.

'It was shocking to me to see the significant amount of Earth that fell into the ocean,' Levin told ABC7 after one incident of flooding last month. 'There would be huge clumps, up to five feet. That's a lot of feet.'

A 2018 report by the Union of Concerned Scientists said more than 300,000 American homes, with a collective value of nearly $120 billion, are at risk of 'chronic inundation' by water within 27 years. 

It is little wonder then that residents in the beachside town of Wells, Maine, are also panicking. 

The state has been particularly hard-hit by flooding this January with a record 14.57-foot high tide measured in parts and many citizens evacuated.

A Facebook account belonging to the Town of Wells, Maine, wrote: 'In the aftermath of the recent devastating storm that swept through our coastal community, we are immensely grateful for the resilience and understanding displayed by our residents. 

Experts say beachfront properties are threatened by a perfect storm of increased cliff erosion, rising sea levels and extreme weather events. Pictured: homes on the coastline in Narragansett, Rhode Island
+15
View gallery
Experts say beachfront properties are threatened by a perfect storm of increased cliff erosion, rising sea levels and extreme weather events. Pictured: homes on the coastline in Narragansett, Rhode Island

A report published last year by the National Ocean Service said sea levels around the country will rise by 10 to 12 inches on average by 2050, primarily due to climate change. Pictured: homes along the Rhode Island coastline
+15
View gallery
A report published last year by the National Ocean Service said sea levels around the country will rise by 10 to 12 inches on average by 2050, primarily due to climate change. Pictured: homes along the Rhode Island coastline

Maine was particularly hard-hit by flooding in January with a record 14.57-foot high tide measured in some parts and many citizens evacuated. Pictured: the beachside town of Wells, Maine this month
+15
View gallery
Maine was particularly hard-hit by flooding in January with a record 14.57-foot high tide measured in some parts and many citizens evacuated. Pictured: the beachside town of Wells, Maine this month

A Facebook account belonging to the Town of Wells, Maine, wrote: 'In the aftermath of the recent devastating storm that swept through our coastal community, we are immensely grateful for the resilience and understanding displayed by our residents.' Pictured: the coastline in Wells, Maine on January 10
+15
View gallery
A Facebook account belonging to the Town of Wells, Maine, wrote: 'In the aftermath of the recent devastating storm that swept through our coastal community, we are immensely grateful for the resilience and understanding displayed by our residents.' Pictured: the coastline in Wells, Maine on January 10 

Extreme climate events caused upwards of $92.9 billion in damage in 2023, figures from the National Centers for Environmental Information show. Pictured: Wells, Maine in January
+15
View gallery
Extreme climate events caused upwards of $92.9 billion in damage in 2023, figures from the National Centers for Environmental Information show. Pictured: Wells, Maine in January

Large swathes of the country have been hit with record flooding and tornadic storms in recent weeks, with images of homes submerged in water plastered across the news. Pictured: Wells, Maine after severe flooding
+15
View gallery
Large swathes of the country have been hit with record flooding and tornadic storms in recent weeks, with images of homes submerged in water plastered across the news. Pictured: Wells, Maine after severe flooding

'The challenges posed by coastal flooding have left their mark, but your cooperation, and the dedication of our municipal employees shines through.'

Meanwhile Narragansett, Rhode Island, was hit by back-to-back storms earlier this month. Video footage shows homes on the town's Roger Wheeler Beach on the verge of collapse.

Experts say beachfront properties are threatened by a perfect storm of increased cliff erosion, rising sea levels and extreme weather events. 

A report published last year by the National Ocean Service said sea levels around the country will rise by 10 to 12 inches on average by 2050, primarily due to climate change.

The numbers might seem small, but every inch raises the risk of devastating flooding in coastal areas as well as exacerbating erosion.

Homes in Narragansett, Rhode Island, pictured, were hit by two back-to-back storms last week
+15
View gallery
Homes in Narragansett, Rhode Island, pictured, were hit by two back-to-back storms last week

Every inch by which sea levels rise increases the risk of devastating flooding in coastal areas. Pictured: a crumbling cliff edge in Narragansett, Rhode Island
+15
View gallery
Every inch by which sea levels rise increases the risk of devastating flooding in coastal areas. Pictured: a crumbling cliff edge in Narragansett, Rhode Island

The Federal Emergency Management Agency can pay owners to leave flood prone properties but they only qualify if they fell victim to a presidentially-declared disaster operates a similar program which pays owners to leave flood-prone properties
+15
View gallery
The Federal Emergency Management Agency can pay owners to leave flood prone properties but they only qualify if they fell victim to a presidentially-declared disaster operates a similar program which pays owners to leave flood-prone properties

Compounding the problem is the fact that many insurance policies will not cover damage incurred from erosion.

Extreme climate events caused upwards of $92.9 billion in damage in 2023, figures from the National Centers for Environmental Information show.

But standard policies will have specific 'named exclusions' such as floods, earthquakes, landslides, sinkholes and earth movement. The latter is the category under which erosion falls.

Owners must also take out flood insurance separately which again does not cover cliff erosion. The average cost of flood insurance for those in a 'high risk' area is $1,607 a year, according to analysis by PolicyGenius.

Politicians have attempted to find a solution to the problem. In 2021, California officials floated the idea of buying back coastal properties from their owners and renting them out until they're too dangerous to live in. 

A photo released by the Nantucket Natural Resources Department shows properties sitting precariously on eroded cliffs
+15
View gallery
A photo released by the Nantucket Natural Resources Department shows properties sitting precariously on eroded cliffs

A group of homeowners in Nantucket recently lost a battle to keep a self-funded 'shield' they erected to keep homes safe from the sea. Pictured: a flooded home on the island
+15
View gallery
A group of homeowners in Nantucket recently lost a battle to keep a self-funded 'shield' they erected to keep homes safe from the sea. Pictured: a flooded home on the island

The island, located near Cape Cod, was also hit hard by recent coastal flooding
+15
View gallery
The island, located near Cape Cod, was also hit hard by recent coastal flooding

And the Federal Emergency Management Agency already operates a similar program which pays residents to leave flood-prone properties.

However Americans only qualify if their home fell victim to a presidentially declared disaster.

Some residents are beginning to fight back. On the island of Nantucket, Massachusetts, a group of wealthy homeowners attempted to take the issue into their own hands by erecting a huge 'shield' of plastic fiber tubes filled with hardening sand slurry.

The barrier stretched across 900 feet of the beach to protect properties from the water. But the privately funded project, which began 10 years ago and is thought to have cost $10 million, ignited a bitter rift in the community with conservations claiming it could impede the beach's natural erosion project.

This month, the group agreed to dismantle the project after an order from town officials. ''', '''Weather tracker: Tropical Cyclone Kirrily brings 170km/h gusts to Queensland
Faye Hulton (Metdesk)
Cyclone downgraded after dense fog hits parts of the US and India while drought affects Philippines and southern Africa

Fri 26 Jan 2024 04.57 EST
7
Tropical Cyclone Kirrily made landfall on the coast of Queensland on Thursday night (local time). Kirrily originated as a tropical low over the Coral Sea, and gradually intensified over several days. The tropical cyclone then quickly intensified on Thursday, reaching a category 2 system by 10am AEST, and category 3 by 3pm, producing gusts of 170 km/h (105mph). As Kirrily moved inland five hours later, it left more than 34,000 homes and business without power in Townsville. However, the cyclone was quickly downgraded back to a category 1 by midnight.

Earlier in the week, dense fog developed from Montana all the way south to the Gulf of Mexico, reducing visibility on Tuesday to less than a quarter mile for many. The combination of last week’s arctic blast, followed by the introduction of warmer air from the south this week, allowed water vapour to condense closer to the surface, which is also known as advection fog. Dense fog reappeared on Thursday morning, affecting just under 99 million people from North Dakota across to central Pennsylvania, and as far south as New Orleans.

Satellite map of Tropical Cyclone Kirrily
‘Like a ghost howling’: ex-Tropical Cyclone Kirrily to bring heavy rain to Queensland as Townsville residents survey damage
Read more
Northern India has also been experiencing dense fog over the past few days, with visibility falling to zero in many places on Wednesday. As a consequence, the Indian Meteorological Department issued a red fog warning on Wednesday morning for Uttar Pradesh, Punjab, Harana, Chandigarh and Delhi, which is valid until 28 January. Unlike in the US, it is the warmer, sunnier days, followed by cooled nights, which allows water fog to form, also known as radiation fog.

Meanwhile, in the Philippines, PAGASA (The Philippine Atmospheric, Geophysical and Astronomical Services Administration) has declared a drought across eight provinces on Luzon Island. Many other provinces, notably Metro Manila, in which the capital city resides, have also been experiencing prolonged dry spells. A drought is here characterised by five consecutive months of below average rainfall, while a dry spell is after only three months. El Niño has been linked to the reduced rainfall, which is set to continue until at least the end of spring.

Other regions have also been experiencing dry spells linked to El Niño in recent months. The World Food Programme has suggested that by the end of January, many southern African nations including Namibia, Botswana, Zimbabwe, Mozambique and southern Madagascar, will have received below average rainfall based on a combination of observations and forecasts. Consequently, the Meteorological Services Department of Zimbabwe has advised farmers to enhance water collection and harvesting when possible.''', '''eff Bezos Got Beyoncé to Come to His Birthday Party?
By Emily Leibert, a night blogger for the Cut 

Photo-Illustration: by The Cut; Photos: Getty Images
Big space guy Jeff Bezos and fiancée Lauren Sánchez have been known to collect celebrities like little trophies, not unlike how cats deposit mouse-shaped toys around the house as evidence of the hunt. At their first engagement party, the couple clinked Champagne flutes with a disturbingly monied crew including Leonardo DiCaprio, Bill Gates, and Queen Rania of Jordan. At their second engagement party — a union so nice, they forced their friends to celebrate twice! — they roped the Kar-Jenners, Scooter Braun, and Robert Pattinson into the same room. And at Bezos’s 60th birthday party on Saturday night, event planners reportedly wrangled yet another queen into attendance.

In no particular order, here is Bezos’s stacked birthday guest list, according to “Page Six”: Ivanka Trump, Jared Kushner, Russell Wilson, Oprah, Gayle King, Kim Kardashian, Hailey Bieber, Bob Iger, Kendall Jenner, Roger Goodell, and … Jay-Z and Beyoncé?


As of Wednesday night, there are no public photos of Beyoncé in attendance, but there are two notable intersections of the disparate Bezos and Knowles-Carter universes that might explain the reported appearance. For one, Bezos, Sanchéz, and the Kardashian clan were spotted together at the Renaissance Tour in Los Angeles in September. So maybe Bezos is just a fanboy who shot his shot in the singer’s DMs. Then again, there were also reports back in 2021 that Bey and Jay vacationed on a 450-foot megayacht (which is longer than a football field, in case you were wondering) belonging to Bezos. Perhaps they’re genuine friends, or perhaps they just get yacht-y together from time to time. That’s what rich people do for fun, right?

While the Bey-zos connection remains a mystery, Alive Boy appeared to have a great night getting serenaded by close pals Katy Perry, Super Bowl–bound Usher, and the Black Eyed Peas. McDonald’s was served, as was caviar. Sánchez also reportedly wore a $4,995 Judith Leiber rocket-ship purse because, well, you know how Jeff loves a testosterone-fueled space race. All of this, and we’ve still got a wedding to go. Can’t wait!''', '''A “Feud” Between Beyoncé And Janet Jackson? Say It Ain’t So!
Jessica Bennett
Wed, January 24, 2024 at 7:31 AM PST·3 min read
38


You’re a beautiful liar if you say Beyoncé and Janet Jackson are actually beefing, as Bey’s mother, Tina Knowles, has explained why she liked a social media post bashing the “Rhythm Nation” icon.

According to Knowles, liking the post was accidental, she explained on Instagram after fans took the gesture as disrespectful to Ms. Jackson (if you’re nasty).

“I am saddened by this. Janet Jackson is an icon and I would never question another artist. I took Destiny’s Child to see Janet when they were 15 years old. They always looked up to her and she opened doors for Beyoncé,” she captioned the IG video.

“Why would I hate on her ? I know now not to be in a rush and to be very careful. (I learned my lesson) ! Especially when I have so much going on.. I just wanted to clarify. . I do know that you can see people’s likes so I would be pretty dumb to do that.”

Check out Ms. Tina explaining herself below.


The original post found its creator complaining of ticket prices for the second leg of Jackson’s Together Again Tour, hitting the road summer 2024. They also compared her stage show to that of Bey’s on her Renaissance World Tour.

“What a joke….they add $1000 for a 15 second meet & greet again. No front row for fans unless you want to bend over so they can continue to rake in the cash. You want Beyoncé prices, have real production. Just disgusted. I’m not the only one…oh the DMs I’m getting. Way to thank your fans for years of support,” Instagram user @let68 captioned the post.

Fans of the youngest Jackson immediately took notice of Knowles liking the post, with one responding, “I do not understand what is the weird obsession with disrespecting The Jacksons. To see Tina Knowles like a post that disrespect Janet makes me lose all respect for her.”

Another observer on Twitter added, “Like out of all people, yall choose THEE Janet Jackson to mess with? She’s literally the most unproblematic popstar ever. The woman who paved the way Beyonce walked through years later. Have some shame and respect.”


Fortunately, the 70-year-old‘s explanation was enough for most who also didn’t want to see the two divas and their fanbases beef. One Beyoncé fan page also shared past footage of the two singers heaping praise on one another to further squash the idea of their being an issue between them.


Neither Beyoncé nor Jackson have commented on the social media post or alleged “feud.”''', '''Since the inception of social media, stan culture has had a sizeable impact on the music industry. Whether it be Rihanna’s Navy, Nicki Minaj’s notorious Barbz or Beyoncé’s BeyHive, fans have found a way to create a community on various sites.

Amidst the heavy discourse around Megan Thee Stallion and Nicki Minaj, fans made Beyoncé trend on Twitter because she does not “embarrass” her fan base. The conversation started when a fan tweeted, “Love you down for never talking to us, mamas, Beyoncé, never embarrassed.” The tweet seemingly plays on the idea that the Renaissance songstress doesn’t speak to fans unless she has something to sell.


One fan commented on the post saying, “Love her for only coming to y’all when she needs money from y’all? She hasn’t even given y’all visuals, and it’s been d**n near a year since she released the trailer. I’d actually be pissed that y’all have to rely on the element of surprise.”


While another mentioned, “And y’all love to overlook her & HER FAMILY, the nasty BEHIND THE SCENES [things and] the things that have been done to GET HER WHERE SHE IS. Wanna talk about morals?! [Laughing my a** off]! You pick and choose. Y’all just don’t like Nicki. Next case.”


One supporter pointed to Bey’s mother, Tina Knowles, as someone who serves as a spokesperson for her daughter. The Twitter user said, “Ms. Tina is bridging the gap for us. She gives the little nuggets we need.”


Unlike most stars in music, Beyoncé rarely uses her social media accounts to voice her opinion on anything. Instead, she shares photos or video content that speaks for itself. Stars like Minaj and Cardi B are much more active and utilize features like Instagram Live, Twitter and other apps to connect with their fans in a way that works for them.''', '''Beyoncé's Mom Apologizes for 'Disrespecting' Janet Jackson
Jan 24, 2024 at 7:33 AM EST
AD Loading ...

Video Player is loading.
By Shannon Power
Pop Culture & Entertainment Reporter
FOLLOW
1
Beyoncé's mom found herself in hot water and needing to apologize to Janet Jackson fans after she seemed to disrespect the music star.

The furor started when Tina Knowles liked an Instagram post by a friend that complained about the ticket prices for Jackson's upcoming tour.

"What a joke....they add $1000 for a 15 second meet & greet again. No front row for fans unless you want to bend over so they can continue to rake in the cash," Instagram user @let68 captioned the post.

tina knowles and janet jackson
[MAIN IMAGE] Tina Knowles attends DIVAS Simply Singing! on November 19, 2023 in Los Angeles, California. [INSET IMAGE] Janet Jackson attends the Christian Siriano SS24 Runway Show on September 08, 2023 in New York City.... More
RODIN ECKENROTH/JAMIE MCCARTHY/GETTY IMAGES
They continued by arguing to charge a lot of money for tickets you need to have a Beyoncé level of production, referring to the singer's recent Renaissance world tour.

"You want Beyoncé prices, have real production. Just disgusted. I'm not the only one...oh the DMs I'm getting. Way to thank your fans for years of support," @let68 wrote.

Sign up for Newsletter
NEWSLETTER
The Bulletin
Your daily briefing of everything you need to know
Email address
 
By clicking on SIGN ME UP, you agree to Newsweek's Terms of Use & Privacy Policy. You may unsubscribe at any time.
Some people noticed Knowles had liked the post and took it as a sign she was being negative towards Jackson. A number of people also hit out at Beyoncé's fans for joining a social media pile on about Jackson.

"Tina Knowles responds about her shading Janet Jackson. I'm hoping Beyoncé fans can leave Janet alone I love Beyoncé too but Yall doing the most dragging Janet," one person wrote on X, formerly Twitter.

Another added: "there's no way i opened this app to janet jackson slander. the disrespect... get a grip."

"I do not understand what is the weird obsession with disrespecting The Jacksons. To see Tina Knowles like a post that disrespect Janet makes me loose all respect for her," a third said.

Newsweek contacted Knowles and Jackson by email for comment.

READ MORE
Taylor Swift's subtle nod to Beyoncé
Donald Trump goes after Taylor Swift
Beyoncé was called the "Queen of Rap" and the internet lost it
Knowles later took to her own Instagram to clarify the situation and said she had nothing but love for Jackson.

"I am saddened by this. Janet Jackson is an icon and I would never question another artist. I took Destiny's Child to see Janet when they were 15 years old. They always looked up to her and she opened doors for Beyoncé," she captioned the video.

"Why would I hate on her ? I know now not to be in a rush and to be very careful. (I learned my lesson) ! Especially when I have so much going on.. I just wanted to clarify. . I do know that you can see people's likes so I would be pretty dumb to do that."

In the video, Knowles explained she sometimes likes posts on Instagram without paying direct attention to what they're saying, but vowed to be more careful in the future.

"First of all, being on the other side of that, I know better than anyone what a great production cost and that is expensive. And I would never criticize another artist, let alone Janet Jackson, who is the queen of production," Knowles began.

"And her family [The Jacksons]? I mean they are the First Family of music always have been, and will always will be in our love and I respect them.

"Secondly, I stay away from negativity. If you follow me then you know that I'm not involved in the mess. What I am guilty of is liking post, going through when I'm in a big hurry and I don't have time to really read and liking things because I trust that these are people that I follow, that was a big mistake."

Knowles concluded: "I will never do that again. So I just want you to hear it from my mouth that I would absolutely never do that."

Jackson is currently completing her first tour in four years called Janet Jackson: Together Again. It was meant to mark her 50th anniversary of being in the music industry but was postponed in 2020 due to the coronavirus pandemic.''', '''Campaign launched to aid farmers still reeling from 2023’s extreme weather
A coalition of organizations aims to raise $20 million from residents, visitors and college alumni to support Vermont agriculture.
By K. Fiegenbaum
January 25, 2024, 4:54 pm
Click to share on Facebook (Opens in new window)Click to share on X (Opens in new window)Click to email a link to a friend (Opens in new window)Click to share on LinkedIn (Opens in new window)
an aerial view of a flooded farm.
Aerial photo of Allen Bros. in Westminster, Vt., and the surrounding area of Route 5 that was flooded as the Connecticut River crested on July 11, 2023. Kristopher Radder/The Brattleboro Reformer via AP
A diverse group of organizations is working with the Vermont Agency of Agriculture, Food & Markets to raise money for farmers affected by last year’s extreme weather — from late spring’s hard frost to the summer’s flooding.

The “Dig Deep Vermont” campaign, announced last week, aims to raise $20 million to provide relief for more than 350 farms across the state. However, a press release announcing the effort noted that the need is estimated to be far greater: $45 million, according to the state Agency of Agriculture.

Threats to farm solvency include crop damage, inadequate winter feed supplies for animals, financial ruin from lost harvests needed to pay loans, and irreparable damage to equipment and facilities, the press release said. According to the campaign’s website, many Vermont farms are at risk of long-term devastating loss or, in some cases, failing altogether.

Sign up for Final Reading, our inside line to what's happening at the Statehouse. Delivered to your inbox Tuesday through Friday during the Vermont legislative session.

Email
Enter your email
two people standing in the doorway of a room.
“Vermont’s farms faced significant challenges this year from freezing to floods,” Gov. Phil Scott said in the press release. “The loss is significant, and farmers need our help. We must work to support our ag economy and the local economies in every Vermont county.”

Organizations helping lead the campaign include the Vermont Association of Broadcasters, Ski Vermont and the Association of Vermont Credit Unions as well as an alliance of local and out-of-state private enterprises.

The Vermont Farm Bureau will distribute the donations until the campaign concludes late this summer. Farmers seeking assistance should fill out a Dig Deep Vermont Farm Assistance Request Form.''', '''Pudu Robotics CEO predicts that service robot market will expand
By Eugene Demaitre | January 27, 2024
Listen to this article
Voiced by Amazon Polly
FacebookTwitterLinkedInRedditPinterestShare
Parkhotel employees in Eisenstadt, Austria, celebrate the arrival of Pudu's service robots.
Parkhotel employees in Eisenstadt, Austria, celebrate the arrival of service robots. Source: Pudu Robotics

Commercial service robots are more common in East Asia than elsewhere, but the rest of the world could catch up in 2024, according to Pudu Technology Co. The Shenzhen, China-based company claimed that it is China’s top exporter of such robots.

“If 2023 was the year of GenAI, I believe 2024 will be the year of the robot,” stated Felix Zhang, founder and CEO of Pudu Robotics. “While humanoid robots and food-making robots grabbed headlines in 2023, the untold story is that it’s the humble service robot — robots that skillfully deliver items and clean floors, often in high-traffic areas — that are actually ready to scale in 2024.”

Last year, Pudu Robotics said it deployed robots across 600 cities in 60 countries. The company also partnered with SoftBank and Nippon Otis Elevators, opened its autonomous mobile robot (AMR) management platform to developers, and won Red Dot and iF Design awards. In addition, it raised more than $15 million in Series C3 funding.

Zhang discussed Pudu’s current offerings and his outlook for this year with The Robot Report:

Service robots to take on more healthcare roles
You have predicted more robots in hospitals and senior living facilities. Does Pudu offer robotics specifically for elder care?

Zhang: Pudu Robotics offers several robots that are deployed in senior living facilities to assist facility staff and residents in their day-to-day tasks and improve the emotional well-being of the elderly. The robots include the BellaBot and KettyBot, two models of delivery robots that can serve food or medicine, assist with returning items, and in some cases even interact with residents.

In addition, although it’s not designed to interact with residents, the PUDU CC1 cleaning robot can help keep senior living facilities tidy, as it is designed to scrub, sweep, vacuum and mop in care homes and other commercial settings. These capabilities automate menial tasks for overwhelmed workers and set the standard for hygiene in autonomous cleaning.

For example, a chain elderly care institution in Hong Kong, which operates 12 nursing homes with 1,600 beds, has adopted CC1 for cleaning the internal environment, reducing the workload of the staff.

PUDU’s robots are especially timely, as more than 1 in 6 Americans are now 65 years or older, and life expectancy continues to grow, thanks to medical advancements. Our aging population is contributing to a major healthcare staffing crisis.

The next 12 months will see the increased adoption of robots in healthcare, as short-staffed senior-living facilities employ the technology to complete tasks. They can monitor daily routines, provide reminders for medication schedules, detect changes in body temperature, and warn medical professionals and families of any abnormalities. 

Robots can also provide emotional support for the elderly and robot-assisted living will become a crucial asset for the growing elderly population. Loneliness is a common problem for many older people, robots provide company and can engage in activities such as communication, storytelling, and playing music.

The traditional way of caring for the elderly often falls short of meeting all of their needs, and robots are able to fill the gaps.


Pudu addresses global markets, economic headwinds
How is the U.S. market for service robots growing in comparison with other regions?

Zhang: The global market for service robots is soaring, and the U.S. is beginning to catch up with its peers in Asia. According to the International Federation of Robotics [IFR], a non-profit industry association, sales of robots used in the service industry grew by 37% worldwide in 2022.

In 2024, the U.S. market is expected to generate the most revenue in the service robot industry, but regions like Japan are leading the way in development and adoption of the technology. In many developing countries, the service industry is hobbled by ever-mounting challenges in hiring workers.

In response, Pudu Robotics has engaged in a massive expansion beyond the borders of its home market since 2020, achieving rapid growth in shipments. Pudu leads the global market as China’s No. 1 service robot exporter, and cumulative global shipments are over 70,000 units. 

In the U.S., employers facing staff shortages have turned to commercial service robots to provide relief for their remaining workers. Quick-service restaurants [QSRs], for example, expect 51% of tasks to be automated by 2025, while full-service restaurants expect to automate 27% of tasks. Service robots are automating menial tasks, improving overall efficiency, and preventing burnout among their human colleagues. 

While the challenge of labor shortages is universal, how will robotics adoption overcome current economic headwinds?

Zhang: Currently, there are 4 million more open jobs than there are available workers in the U.S. to fill them. As society’s tolerance and acceptance of new technology grows, robots will plug this hole.

U.S. restaurants are a prime example. Owners face a “perfect storm” of an aging population, soaring child-care costs that shrink the pool of available workers, and a pandemic that pointed many workers towards more stable careers.

While economic headwinds may cause some delay, the world is still turning towards an increasingly automated future. Robots are the long-term solution for massive problems facing several industries.


Integration and AI to make robots more useful
From hospitality to healthcare and retail, which areas have the most demand? How much integration will be necessary?

Zhang: We’re seeing an increase in demand for service robots most from the restaurant industry, followed by hospitality, healthcare, and retail by order of demand increase. To meet that global demand, Pudu Robotics is building two new factories near Shanghai that will triple the company’s annual capacity.

Businesses across all four categories are finding it hard to maintain adequate service levels due to staffs being stretched thin. Integration will happen across these industries, as they all are dealing with the effects of the labor shortage. Business will still need human workers, but robots can supplement and improve efficiencies. 

While large language models (LLMs) are improving human-machine interactions, how will they be instantiated in robots rather than on tablets and phones?

Zhang: Large language models can effectively enhance human-robot interaction, particularly in semantic understanding. Take, for instance, a robot serving as a shopping guide in a supermarket.

Previously, to find a specific brand of electric toothbrushes, customers needed to navigate through “personal care,” then “toothbrushes,” followed by “electric toothbrushes,” and finally the brand. With significant improvements in voice recognition and semantic understanding, it’s now possible to locate the item directly through a single command. 

LLMs are highly beneficial for advancing end-to-end algorithms in modules such as positioning, navigation, and perception, significantly enhancing the efficiency of their evolution to achieve global optimization. Robots and LLM integration is already under way, but manufacturers need to ensure that the generated content aligns with human values and safety standards, while also ensuring that robots can reliably and responsibly interact with the real world.

In 2024, these models will be used effectively in robots — as well as tablets and phones — as robots with AI voice interactions and eye-catching displays make use of the advancing technology.''',
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

    # Print the results
    print("Nearest neighbors:")
    for i, index in enumerate(indices[0]):
        print(f"Rank {i+1}: Index {index}, Distance: {distances[0][i]}")



    final_news_array = []
    for i in range(k):
        final_news_array.append(arr[indices[0][i]])


    summary = ''
    for i in range(len(final_news_array)):
        prompt =f"""
        Your task is to extract relevant information from a text on the page of this news article. This information will be used to create a summary.
        Be sure to preserve the important details.
        Text: ```{i}'. '{final_news_array[i]}```
        """
        try:
            response = get_completion(prompt)
        except:
            response = get_completion(prompt)
        print(response)
        summary= summary+' ' +response +'\n\n'
    
    if length == 1:
        num_of_words = 60
    elif length == 5:
        num_of_words = 200
    else:
        num_of_words = 100
    prompt_summary2 =f"""
    There are articles summarized in below text. You are a news correspondent and are required to narrate the most salient points of the news in a paragraph form in '{num_of_words}'. Create the script for yourself. Don't enumerate the articles. Make the flow sound more natural, however, do make the shift from one article to another explicit by using filler phrases like "in another news", "here is some more trump news" etc. The start of the summary should be natural and shouldn't use filler word. Absolutely neccessary to keep the full summary within the word limit on '{num_of_words}'.
    Summary: ```{summary}```
    """
    summary_2 = get_completion(prompt_summary2)
    print("Here is the final narration:",summary_2)
    
    title_prompt =f"""
    Below is a news summary. Please generate a catchy 4 - 5 words headline for the summary.
    Summary: ```{summary_2}```
    """
    title = get_completion(title_prompt)
    print("Here is the title:",title)

#     # The text that you want to convert to audio 
#     mytext = summary_2
    
#     # Language in which you want to convert 
#     language = 'en'
    
#     # Passing the text and language to the engine,  
#     # here we have marked slow=False. Which tells  
#     # the module that the converted audio should  
#     # have a high speed 
#     myobj = gTTS(text=mytext, lang=language, slow=False) 
    
#     # Saving the converted audio in a mp3 file named 
#     # welcome  
#     myobj.save("newssummary_t2.mp3")
    
    elevenlabs.set_api_key("a5b1689de55886b8285a34e04ef7dafa")

    voice = elevenlabs.Voice(
        voice_id = "5Q0t7uMcjvnagumLfvZi",
        settings = elevenlabs.VoiceSettings(
            stability = 0,
            similarity_boost = 0.75
        )
    )

    app.config["NEWS_SUMMARY"] = summary_2
    app.config["TITLE"] = title
    
    if len(summary_2)>2500:
        summary_2 = summary_2[:2400]

    audio = elevenlabs.generate(
        text = summary_2,
        voice = "Paul"
    )
    

    elevenlabs.save(audio, "audio_2.mp3")


    # Return a response
    path_to_mp3 = 'audio_2.mp3'
    return send_file(path_to_mp3, as_attachment=True)

    

@app.route('/get_title')
def text():
    return jsonify({"summary": app.config["NEWS_SUMMARY"], "title": app.config["TITLE"]})
    # return (news_summary,news_title

@app.route('/get_mp3')
def get_mp3():
    # SHUBHAM'S CODE HERE #
    path_to_mp3 = 'static/example.mp3'
    return send_file(path_to_mp3, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)