from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """### The Future of Renewable Energy


Renewable energy has become one of the most important topics of the twenty-first century as countries around the world seek sustainable alternatives to fossil fuels. Traditional energy sources such as coal, oil, and natural gas have powered industries and economies for decades, but they also contribute significantly to greenhouse gas emissions and climate change. Renewable energy sources, including solar, wind, hydroelectric, geothermal, and biomass energy, offer cleaner and more sustainable solutions. Advances in technology have significantly reduced the cost of renewable energy production, making it increasingly competitive with conventional energy sources. Governments, businesses, and researchers are investing heavily in renewable energy infrastructure to meet growing energy demands while minimizing environmental impact. The transition toward renewable energy not only helps reduce carbon emissions but also creates new jobs, stimulates economic growth, and enhances energy security for nations worldwide.

Solar energy has emerged as one of the fastest-growing renewable energy technologies due to its accessibility and declining installation costs. Solar panels convert sunlight directly into electricity using photovoltaic cells, allowing homes, businesses, and utility companies to generate clean power. Large-scale solar farms are being developed in regions with abundant sunlight, while rooftop solar systems enable individuals to produce their own electricity and reduce dependence on centralized power grids. Technological improvements have increased the efficiency of solar panels, allowing them to generate more electricity from the same amount of sunlight. In addition, advances in battery storage systems have addressed one of the primary challenges of solar power: its intermittent nature. By storing excess energy generated during sunny periods, batteries enable a more reliable and consistent energy supply even when sunlight is unavailable.

Wind energy has also become a major contributor to global renewable energy production. Wind turbines harness the kinetic energy of moving air and convert it into electricity through rotating blades connected to generators. Modern wind farms can consist of hundreds of turbines located on land or offshore, where wind speeds are often stronger and more consistent. Offshore wind energy, in particular, has gained attention due to its high energy generation potential and reduced land-use requirements. Engineers continue to design larger and more efficient turbines capable of producing greater amounts of electricity while reducing maintenance costs. Despite challenges such as visual impact, noise concerns, and wildlife interactions, wind energy remains one of the most cost-effective and environmentally friendly sources of electricity available today.

The integration of renewable energy into existing power grids presents both opportunities and challenges. Unlike traditional power plants that can generate electricity continuously, many renewable energy sources depend on weather conditions and natural cycles. As a result, power grid operators must develop advanced systems to balance energy supply and demand effectively. Smart grid technologies, artificial intelligence, predictive analytics, and energy storage solutions are playing increasingly important roles in managing renewable energy resources. Governments are also implementing policies and incentives to encourage renewable energy adoption, including tax credits, subsidies, renewable portfolio standards, and carbon pricing mechanisms. International cooperation and investment in research and development are essential for overcoming technical barriers and accelerating the global transition to clean energy systems.

Looking ahead, renewable energy is expected to play a central role in achieving global sustainability goals and combating climate change. Continued innovation in energy generation, storage, transmission, and efficiency technologies will further improve the viability of renewable energy solutions. Emerging developments such as green hydrogen production, advanced battery technologies, floating offshore wind farms, and next-generation solar materials hold significant promise for expanding renewable energy capabilities. As public awareness of environmental issues continues to grow, consumers and businesses are increasingly choosing cleaner energy options. The widespread adoption of renewable energy has the potential to transform economies, improve public health, reduce environmental degradation, and create a more sustainable future for generations to come. The success of this transition will depend on collaboration among governments, industries, researchers, and communities working together toward a shared vision of a cleaner and more resilient energy future.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap = 0,
)

# split_text and split_documents u can use based on req.

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks[0])