from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation")

model = ChatHuggingFace(llm = llm)


class review(BaseModel):
    name: str = Field(description="Name of the device")
    make: int = Field(description="The year in which this product was launched")
    pros: Optional[list[str]] = Field(default=None,description="a list of pros of this device if any")
    cons: Optional[list[str]] = Field(default=None,description="a list of cons of this device if any")
    sentiment: Literal['pos','neg'] = Field(description="'pos' for positive sentiment and 'neg' for negative...")


parser = PydanticOutputParser(pydantic_object = review)

chatTemplate = ChatPromptTemplate([
    ('system', "You are a review summarizer \n {format_instructions}"),
    ('human', "review is {review}")
], partial_variables = {'format_instructions':parser.get_format_instructions()})


chain = chatTemplate | model | parser


result = chain.invoke({
    "review": """
    Design and Build Quality

One of the standout aspects of the Galaxy S24 is its premium design. The device features:

Elegant and modern aesthetics
Lightweight and comfortable form factor
Durable Armor Aluminum frame
Corning Gorilla Glass protection
Premium in-hand feel
IP68 water and dust resistance

The compact size makes it easy to use with one hand while still offering a large and immersive display experience.

Stunning Display

Samsung continues to lead the industry in display technology, and the Galaxy S24 is no exception.

Key Highlights
Dynamic AMOLED 2X display
Full HD+ resolution
Adaptive 1Hz to 120Hz refresh rate
Excellent color accuracy
Deep blacks and vibrant colors
Outstanding brightness for outdoor visibility
Smooth scrolling and gaming experience

Whether you're watching videos, browsing social media, reading articles, or gaming, the display consistently delivers a premium visual experience.

Flagship Performance

The Galaxy S24 offers flagship-level performance that handles virtually any task with ease.

Performance Benefits
Fast app launches
Smooth multitasking
Responsive user interface
Excellent gaming performance
Efficient power management
Reliable day-to-day usage

The phone remains fluid and responsive even under demanding workloads, making it suitable for both casual users and power users.

Galaxy AI Features

One of the most exciting aspects of the Galaxy S24 is the integration of Galaxy AI.

AI-Powered Features
Live Translate
Real-time call translation
Supports multilingual communication
Convenient for travel and international conversations
Chat Assist
Helps refine messages
Improves writing style and tone
Useful for professional and personal communication
Note Assist
Summarizes lengthy notes
Organizes information efficiently
Increases productivity
Circle to Search
Instantly search anything visible on the screen
Simple and intuitive user experience
Generative Editing
Advanced photo editing capabilities
Smart object manipulation
Creative image enhancement tools

These features help make everyday tasks faster, smarter, and more convenient.

Excellent Camera System

Samsung has equipped the Galaxy S24 with a versatile camera setup capable of producing impressive results across different scenarios.

Photography Strengths
Detailed photos
Vibrant colors
Excellent dynamic range
Reliable autofocus
Strong portrait photography
Consistent image processing
Great low-light performance
Zoom Capabilities
High-quality telephoto camera
Sharp zoomed-in shots
Useful for distant subjects
Versatile shooting options
Selfie Camera
Natural skin tones
High detail retention
Great social media-ready images
Excellent video call quality
Impressive Video Recording

The Galaxy S24 performs exceptionally well in video capture.

Video Features
Sharp video quality
Excellent stabilization
Smooth motion capture
Reliable autofocus tracking
High dynamic range recording
Professional-looking results

Whether recording family moments, travel experiences, or content for social media, the phone delivers polished and high-quality videos.

Battery Efficiency

Battery optimization is another strong area of the Galaxy S24.

Benefits
Efficient power consumption
Reliable all-day performance
Smart battery management
Optimized standby time
Consistent endurance throughout the day

The combination of efficient hardware and software optimization contributes to a dependable battery experience.

Software Experience

Samsung's One UI provides a refined and feature-rich software experience.

Advantages
Clean interface
Smooth animations
Extensive customization options
Useful productivity tools
Strong ecosystem integration
Advanced security features
User-friendly navigation

The software feels mature, polished, and highly customizable.

Long-Term Support

Samsung's commitment to software updates is one of the biggest strengths of the Galaxy S24.

Benefits
Multiple years of Android updates
Long-term security patches
Access to new features over time
Improved device longevity
Better overall value

This ensures the phone remains relevant and secure for years.

Connectivity and Features

The Galaxy S24 includes a comprehensive set of premium features:

5G connectivity
Wi-Fi 6E support
Bluetooth connectivity
NFC support
Samsung DeX compatibility
Stereo speakers
In-display fingerprint scanner
Face recognition
Wireless ecosystem integration
Final Verdict

The Samsung Galaxy S24 is an exceptionally polished flagship smartphone that excels in nearly every area. It offers a premium design, beautiful display, powerful performance, versatile cameras, innovative Galaxy AI features, reliable battery life, and long-term software support. Its combination of cutting-edge technology and everyday practicality makes it a highly compelling choice for users seeking a premium Android experience.

Rating: 9.5/10

Best For: Students, professionals, content creators, travelers, mobile gamers, and anyone looking for a compact premium flagship smartphone with advanced AI capabilities and a refined user experience.
    """
})



print(result)