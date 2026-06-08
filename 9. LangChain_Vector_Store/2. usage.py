from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma


load_dotenv()

model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)



from langchain_core.documents import Document

doc1 = Document(
    page_content="""
    Virat Kohli is an Indian cricketer who plays as a right-handed top-order batter.
    He has represented Royal Challengers Bengaluru (RCB) in the Indian Premier League
    since the inaugural season in 2008. Kohli is widely regarded as one of the greatest
    batters in cricket history due to his consistency across formats, ability to chase
    targets, and exceptional fitness levels. He has scored thousands of runs in the IPL,
    won the Orange Cap multiple times, and has been one of the most valuable players
    for RCB. His batting style combines technical excellence with aggressive intent,
    making him a key player in high-pressure matches.
    """,
    metadata={"player": "Virat Kohli", "team": "Royal Challengers Bengaluru"}
)

doc2 = Document(
    page_content="""
    Rajat Patidar is an Indian right-handed batter who plays for Royal Challengers
    Bengaluru in the IPL. Known for his elegant stroke play and ability to accelerate
    the scoring rate, Patidar gained widespread recognition after playing several
    match-winning innings in crucial knockout matches. He is particularly strong
    against spin bowling and is capable of anchoring an innings while maintaining
    a healthy strike rate. His performances have made him a dependable middle-order
    batter and an important asset for RCB.
    """,
    metadata={"player": "Rajat Patidar", "team": "Royal Challengers Bengaluru"}
)

doc3 = Document(
    page_content="""
    Jasprit Bumrah is an Indian fast bowler representing Mumbai Indians in the IPL.
    He is known for his unique bowling action, deadly yorkers, and ability to deliver
    under pressure. Bumrah has been a cornerstone of Mumbai Indians' bowling attack
    and has played a major role in multiple championship-winning campaigns. His skills
    in both the powerplay and death overs make him one of the most feared bowlers in
    T20 cricket. He consistently ranks among the leading wicket-takers in tournaments
    and is regarded as one of the finest fast bowlers of his generation.
    """,
    metadata={"player": "Jasprit Bumrah", "team": "Mumbai Indians"}
)

doc4 = Document(
    page_content="""
    Rohit Sharma is an Indian opening batter and one of the most successful captains
    in IPL history. Representing Mumbai Indians, Rohit has led the franchise to
    multiple IPL titles and established himself as one of the most reliable batters
    in the competition. He is known for his effortless stroke play, ability to score
    big centuries, and calm leadership style. Rohit excels against both pace and spin
    and has contributed significantly to Mumbai Indians' success over the years.
    """,
    metadata={"player": "Rohit Sharma", "team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="""
    MS Dhoni is a legendary Indian wicketkeeper-batter who has captained Chennai Super
    Kings for most of the franchise's history. Renowned for his finishing ability,
    tactical brilliance, and calm demeanor under pressure, Dhoni has guided CSK to
    multiple IPL championships. His quick glove work behind the stumps, leadership,
    and ability to finish close matches have earned him immense respect from fans
    and players worldwide. He remains one of the most iconic figures in cricket.
    """,
    metadata={"player": "MS Dhoni", "team": "Chennai Super Kings"}
)

doc6 = Document(
    page_content="""
    Ruturaj Gaikwad is an Indian opening batter representing Chennai Super Kings.
    He is known for his elegant batting technique, consistency at the top of the
    order, and ability to build long innings. Gaikwad has won the Orange Cap in the
    IPL and has been instrumental in providing strong starts for CSK. His calm
    temperament and wide range of shots allow him to perform effectively against
    both pace and spin bowlers.
    """,
    metadata={"player": "Ruturaj Gaikwad", "team": "Chennai Super Kings"}
)

doc7 = Document(
    page_content="""
    Shubman Gill is an Indian batter and captain of Gujarat Titans. He is regarded
    as one of the most technically gifted young cricketers in the world. Gill's
    ability to play both attacking and anchoring roles makes him a versatile batter.
    He has scored numerous match-winning innings in the IPL and is known for his
    elegant cover drives, consistency, and composure in pressure situations.
    """,
    metadata={"player": "Shubman Gill", "team": "Gujarat Titans"}
)

doc8 = Document(
    page_content="""
    Rashid Khan is an Afghan leg-spin bowler representing Gujarat Titans. He is one
    of the most successful T20 bowlers globally due to his accuracy, variations,
    and ability to take wickets at crucial moments. Apart from his bowling, Rashid
    is also a capable lower-order batter who can score quick runs. His all-round
    contributions have helped his teams win numerous matches across leagues around
    the world.
    """,
    metadata={"player": "Rashid Khan", "team": "Gujarat Titans"}
)

doc9 = Document(
    page_content="""
    KL Rahul is an Indian wicketkeeper-batter known for his stylish batting and
    versatility. Representing Delhi Capitals, Rahul can adapt his approach depending
    on the match situation. He has scored several IPL centuries and consistently
    ranks among the tournament's leading run scorers. His ability to keep wickets
    and bat in different positions adds significant value to his team.
    """,
    metadata={"player": "KL Rahul", "team": "Delhi Capitals"}
)

doc10 = Document(
    page_content="""
    Axar Patel is an Indian all-rounder and captain of Delhi Capitals. He is known
    for his economical left-arm spin bowling, useful lower-order batting, and
    exceptional fielding abilities. Axar has delivered several match-winning
    performances with both bat and ball. His versatility allows team management
    to use him in different roles depending on the match requirements.
    """,
    metadata={"player": "Axar Patel", "team": "Delhi Capitals"}
)

doc11 = Document(
    page_content="""
    Sanju Samson is an Indian wicketkeeper-batter and captain of Rajasthan Royals.
    He is known for his aggressive batting style, elegant stroke play, and ability
    to dominate bowling attacks. Samson has been a consistent performer in the IPL
    and has led Rajasthan Royals in multiple seasons. His leadership and batting
    contributions make him one of the franchise's most important players.
    """,
    metadata={"player": "Sanju Samson", "team": "Rajasthan Royals"}
)

doc12 = Document(
    page_content="""
    Yashasvi Jaiswal is an Indian opening batter representing Rajasthan Royals.
    He is recognized for his fearless approach, attacking intent, and ability to
    score rapidly in powerplay overs. Jaiswal has produced several memorable IPL
    innings and is considered one of the brightest young talents in Indian cricket.
    His aggressive batting style often puts opposition bowlers under pressure.
    """,
    metadata={"player": "Yashasvi Jaiswal", "team": "Rajasthan Royals"}
)

doc13 = Document(
    page_content="""
    Sunil Narine is a West Indian all-rounder who plays for Kolkata Knight Riders.
    Famous for his mystery spin bowling, Narine has been one of the most effective
    bowlers in IPL history. In recent seasons, he has also excelled as an opening
    batter, providing explosive starts for KKR. His dual ability to contribute with
    both bat and ball makes him one of the most valuable T20 players in the world.
    """,
    metadata={"player": "Sunil Narine", "team": "Kolkata Knight Riders"}
)

doc14 = Document(
    page_content="""
    Andre Russell is a West Indian all-rounder representing Kolkata Knight Riders.
    He is renowned for his immense power-hitting, fast bowling, and match-winning
    performances. Russell can change the course of a game within a few overs through
    explosive batting or crucial wickets. His athletic fielding and ability to
    perform under pressure have made him one of the most impactful players in T20
    cricket.
    """,
    metadata={"player": "Andre Russell", "team": "Kolkata Knight Riders"}
)

doc15 = Document(
    page_content="""
    Nicholas Pooran is a left-handed wicketkeeper-batter representing Lucknow Super
    Giants. He is known for his aggressive batting style, ability to hit sixes
    consistently, and effectiveness against spin bowling. Pooran has played several
    match-winning innings in T20 cricket and is regarded as one of the most dangerous
    middle-order batters in the format. His finishing ability makes him a valuable
    asset in high-pressure run chases.
    """,
    metadata={"player": "Nicholas Pooran", "team": "Lucknow Super Giants"}
)

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15]


vector_store = Chroma(
    embedding_function = model,
    persist_directory= "chroma_db",
    collection_name='sample'
)

#vector_store.add_documents(docs)


# result = vector_store.similarity_search(
#     query = "who is dhoni's wife?",k=2
# )

# result = vector_store.similarity_search_with_score(
#     query="",
#     filter={"team": "Chennai Super Kings"}
# )
virat_personal_life = Document(
    page_content="""
    Virat Kohli was born on November 5, 1988, in Delhi, India, to Prem Kohli and
    Saroj Kohli. He grew up in a middle-class family and showed an interest in
    cricket from a very young age. Recognizing his talent, his family enrolled him
    in the West Delhi Cricket Academy when he was a child. Kohli's father played a
    significant role in supporting his cricketing ambitions and often accompanied
    him to practice sessions and matches.

    During his teenage years, Kohli faced a major personal tragedy when his father
    passed away in 2006. Despite the emotional setback, he displayed remarkable
    determination by continuing to play an important domestic cricket match the next
    day. This incident is often cited as an example of his commitment and mental
    strength.

    Outside cricket, Virat Kohli is known for his disciplined lifestyle and strong
    focus on fitness. Over the years, he transformed his approach to nutrition,
    training, and recovery, becoming one of the fittest athletes in international
    cricket. His emphasis on health and fitness has influenced many young cricketers
    in India.

    Virat Kohli married Bollywood actress Anushka Sharma in December 2017 after
    several years of dating. The couple is among the most well-known celebrity pairs
    in India and is often admired for maintaining a balance between their personal
    and professional lives. They have two children and generally keep their family
    life private, occasionally sharing selected moments with fans through social
    media.

    Apart from cricket, Kohli has been involved in various business ventures,
    endorsements, and charitable activities. Through the Virat Kohli Foundation,
    he has supported initiatives related to child welfare, sports development, and
    education. He is also known for his interest in fashion, entrepreneurship, and
    promoting a healthy lifestyle.

    Friends and teammates often describe Kohli as highly competitive, passionate,
    and driven. While he is known for his intensity on the field, he is also
    recognized for his dedication to family, fitness, and personal growth off the
    field.
    """,
    metadata={
        "player": "Virat Kohli",
        "category": "Personal Life",
        "nationality": "Indian",
        "birth_place": "Delhi, India"
    }
)


vector_store.update_document(document_id = "788a6b3e-bb6a-43f1-a7a1-d49e8a5e1606", document=virat_personal_life)


result = vector_store.get(include=["embeddings","documents", "metadatas"])

# vector_store.delete(ids=['02466c3a-3bb5-454d-9d50-09a1ad6e9f08'])

print(result)