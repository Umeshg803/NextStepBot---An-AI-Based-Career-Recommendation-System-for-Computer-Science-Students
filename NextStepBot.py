import numpy as np
import pandas as pd
import pyttsx3
import streamlit as st
from PIL import Image

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")  # Haven't found any native Bengali voice for microsoft sapi5 engine :(
engine.setProperty("voice", voices[0].id)

dataset = pd.read_csv("roo_data.csv")

ANS_QUES_MAP = {
        "Hello There": ["hello", "hi", "hey", "hello there"],
        "I am Fine. What about you?": ["how are you", "how are you doing"],
        "It is nice to hear.": ["fine", "I am also fine", "I am well", "fine thank you", "i am doing well", "pretty good"],
        "I am NextStepBot,I am hear to help you with your carrer,Type Start Test to begain the test": ["who are you", "what is your identity", "what is your name"],
        "Sy MCA Group 1": ["who made you", "who created you", "who is your creator"],
        "You are welcome": ["thanks", "thank you"],
        "Thank you": ["nice", "great", "good", "wonderful"],
        "Should I tell you a joke or play any music for you?": ["my mood is off", "i am not feeling great"]
    }

def output(answers):
    #-----------------------------Processing user Inpute----------------#
    #---------------Applying OneHot & Lable  Encoding-----------#
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder = LabelEncoder()

    #---------------conversion of all categorial column values to vector/numerical--------#
    for i in range(14,38):
        answers[i:] = labelencoder.fit_transform(answers[i:])

    #--------------normalizing the non-categorial column values---------#
    from sklearn.preprocessing import Normalizer
    answers1 = answers[:14]
    # print(answers1) 
    answers1_2d = np.reshape(answers1, (1, -1))
    # print(answers1_2d)
    normalized_data = Normalizer().fit_transform(answers1_2d)
    # print(normalized_data)

    answers2 = answers[14:]
    answers2_2d = np.reshape(answers2, (1, -1))
    # print(answers2_2d.shape)
    dff1 = np.append(normalized_data,answers2_2d,axis=1)
    # dff1.shape

    data = dataset.iloc[:,:-1].values
    label = dataset.iloc[:,-1].values

    #------------------Encoding Final Output column Values------------#
    label = labelencoder.fit_transform(label)
    # print(len(label))
    y=pd.DataFrame(label,columns=["Suggested Job Role"])

    # -----------Manually loading--------------#
    # load the model from disk
    import pickle
    svc_clf = pickle.load(open('newmodels/svc_model.h5', 'rb'))
    #--------doing prediction-----#
    svm_y_pred = svc_clf.predict(dff1)   
    print(svm_y_pred)

    decoded_labels = labelencoder.inverse_transform(svm_y_pred)
    print(decoded_labels)
    return decoded_labels


def main():
    
    def botans(query):
        # basic conversation
        for key, value in ANS_QUES_MAP.items():
            if query in value:
                return key

    def speak(audio):
        """
        Speaks a sentence.
        Param - audio (string)
        """
        if engine._inLoop:
            engine.endLoop()
        engine.say(audio)
        engine.runAndWait()
        # t3 = threading.Thread(target=engine.runAndWait)
        # t3.start()

    def get_text():
        x = st.text_input("You : ")
        x=x.lower()
        return x

    questions=['Acedamic percentage in Operating Systems.', 'Acedamic percentage in Algorithms.',
       'Acedamic Percentage in Programming Concepts.',
       'Acedamic Percentage in Software Engineering.', 'Percentage in Computer Networks.',
       'Acedamic Percentage in Electronics Subjects.',
       'Acedamic Percentage in Computer Architecture.', 'Acedamic Percentage in Mathematics.',
       'Acedamic Percentage in Communication skills', 'How many hours of working you will prefer per day?',
       'How much you will rate yourself in Logical quotient?', 'how many hackathons have you participated?', 'How much you will rate yourself in coding skills?',
       'How much you will rate yourself in public speaking?', 'Can you work long time before system?',
       'Are you capable of self-learning?', 'Have you done any Extra-courses?', 'Have you done any certifications?',
       'Workshops done?', 'Have you taken any talent tests before?', 'Have you taken olympiads test?',
       'How are your reading and writing skills?.', 'what is your memory capability score?',
       'What is your interested subjects?', 'What is your interested career area?', 'What will you prefer Job or Higher Studies?',
       'What type of company do you want to settle in?',
       'Have you Taken any inputs from seniors or elders?', 'Are you interested in games?',
       'Which type of Books are you interested?', 'What matters for you most work or salary?',
       'Are you in a Realtionship?', 'What kind of behaviour you have?',
       'Are you intrested in Management or Technical job?', 'What will you prefer?', 'Are you hard worker or smart worker?',
       'Have you worked in teams ever?', 'Are you Introvert?']
    flag=1
    st.title("NextStepBot")
    banner=Image.open("img/21.png")
    st.image(banner, use_column_width=True)
    st.write("Hi! I'm NextStepBot, your personal career counseling bot. Ask your queries in the text box below and hit enter. If and when you are ready to take our personality test, type \"start my test\" and you're good to go!")


    query = get_text()
    if (query==""):
        ans = "Hi, I'm NextStepBot. \nHow can I help you?"
        st.text_area("NextStepBot:", value=ans, height=100, max_chars=None)
        speak(ans)

    elif 'start test' in query:
        flag=0
        ans = "Sure, good luck!"
    # elif(flag==0):
    #     #st.write(flag)
    #     ans = "Sure, good luck!"
    #     speak(ans)
    else:
        ans = botans(query)
        st.text_area("NextStepBot:", value=ans, height=100, max_chars=None)
        speak(ans)
        
    if(flag==0):
         
        st.title("Job MATCH IQ TEST:")
        kr = st.selectbox("Would you like to begin with the test?", ["Select an Option", "Yes", "No"])
        if (kr == "Yes"):
            
            lis = []
            if (kr == "Yes"):
                st.subheader("Question 1")
                st.write(questions[0])
                inp =st.text_input("",key='1')
                if ((inp != "")):
                    lis.append(int(inp))
                    
                    st.subheader("Question 2")
                    st.write(questions[1])
                    inp2 = st.text_input("",key='2')
                    if (inp2 != ""):
                        lis.append(int(inp2))
                        
                        st.subheader("Question 3")
                        st.write(questions[2])
                        inp3 = st.text_input("",key='3')
                        if (inp3 != ""):
                            lis.append(int(inp3))

                            st.subheader("Question 4")
                            st.write(questions[3])
                            inp4 = st.text_input("", key='4')
                            if (inp4 != ""):
                                lis.append(int(inp4))

                                st.subheader("Question 5")
                                st.write(questions[4])
                                inp5 = st.text_input("", key='5')
                                if (inp5 != ""):
                                    lis.append(int(inp5))

                                    st.subheader("Question 6")
                                    st.write(questions[5])
                                    inp6 = st.text_input("", key='6')
                                    if (inp6 != ""):
                                        lis.append(int(inp6))
                                        print(lis)

                                        st.subheader("Question 7")
                                        st.write(questions[6])
                                        inp7 = st.text_input("", key='7')
                                        if (inp7 != ""):
                                            lis.append(int(inp7))

                                            st.subheader("Question 8")
                                            st.write(questions[7])
                                            inp8 = st.text_input("", key='8')
                                            if (inp8 != ""):
                                                lis.append(int(inp8))

                                                st.subheader("Question 9")
                                                st.write(questions[8])
                                                inp9 = st.text_input("", key='9')
                                                if (inp9 != ""):
                                                    lis.append(int(inp9))

                                                    st.subheader("Question 10")
                                                    st.write(questions[9])
                                                    inp10 = st.selectbox("",["Select an Option",10,9,8,7,6,5,4,3,2,1,0], key='10')
                                                    if (inp10 != "Select an Option"):
                                                        lis.append(int(inp10))

                                                        st.subheader("Question 11")
                                                        st.write(questions[10])
                                                        inp11 = st.selectbox("",["Select an Option",10,9,8,7,6,5,4,3,2,1,0], key='11')
                                                        if (inp11 != "Select an Option"):
                                                            lis.append(int(inp11))

                                                            st.subheader("Question 12")
                                                            st.write(questions[11])
                                                            inp12 = st.selectbox("",["Select an Option",10,9,8,7,6,5,4,3,2,1,0], key='12')
                                                            if (inp12 != "Select an Option"):
                                                                lis.append(int(inp12))

                                                                st.subheader("Question 13")
                                                                st.write(questions[12])
                                                                inp13 = st.selectbox("",["Select an Option",10,9,8,7,6,5,4,3,2,1,0], key='13')
                                                                if (inp13 != "Select an Option"):
                                                                    lis.append(int(inp13))
                                                                    
                                                                    st.subheader("Question 14")
                                                                    st.write(questions[13])
                                                                    inp14 = st.selectbox("",["Select an Option",10,9,8,7,6,5,4,3,2,1,0], key='14')
                                                                    if (inp14 != "Select an Option"):
                                                                        lis.append(int(inp14))                                                                        

                                                                        st.subheader("Question 15")
                                                                        st.write(questions[14])
                                                                        inp15 = st.selectbox("",["Select an Option","yes","no"], key='15')
                                                                        if (inp15 != "Select an Option"):
                                                                            lis.append(inp15)

                                                                            st.subheader("Question 16")
                                                                            st.write(questions[15])
                                                                            inp16 = st.selectbox("",["Select an Option","yes","no"], key='16')
                                                                            if (inp16 != "Select an Option"):
                                                                                lis.append(inp16)

                                                                                st.subheader("Question 17")
                                                                                st.write(questions[16])
                                                                                inp17 = st.selectbox("",["Select an Option","yes","no"], key='17')
                                                                                if (inp17 != "Select an Option"):
                                                                                    lis.append(inp17)

                                                                                    st.subheader("Question 18")
                                                                                    st.write(questions[17])
                                                                                    inp18 = st.selectbox("",["Select an Option","shell programming",
                                                                                                             "machine learning","app development",
                                                                                                             "python","r programming","information security",
                                                                                                             "hadoop","distro making","full stack"], key='18')
                                                                                    if (inp18 != "Select an Option"):
                                                                                        lis.append(inp18)

                                                                                        st.subheader("Question 19")
                                                                                        st.write(questions[18])
                                                                                        inp19 = st.selectbox("",["Select an Option","cloud computing","database security",
                                                                                                                "web technologies","data science",
                                                                                                                "testing","hacking","game development",
                                                                                                                "system designing"], key='19')
                                                                                        if (inp19 != "Select an Option"):
                                                                                            lis.append(inp19)

                                                                                            st.subheader("Question 20")
                                                                                            st.write(questions[19])
                                                                                            inp20 = st.selectbox("",["Select an Option","yes","no"], key='20')
                                                                                            if (inp20 != "Select an Option"):
                                                                                                lis.append(inp20)

                                                                                                st.subheader("Question 21")
                                                                                                st.write(questions[20])
                                                                                                inp21 = st.selectbox("",["Select an Option","yes","no"], key='21')
                                                                                                if (inp21 != "Select an Option"):
                                                                                                    lis.append(inp21)

                                                                                                    st.subheader("Question 22")
                                                                                                    st.write(questions[21])
                                                                                                    inp22 = st.selectbox("",["Select an Option","excellent","medium","poor"], key='22')
                                                                                                    if (inp22 != "Select an Option"):
                                                                                                        lis.append(inp22)

                                                                                                        st.subheader("Question 23")
                                                                                                        st.write(questions[22])
                                                                                                        inp23 = st.selectbox("",["Select an Option","excellent","medium","poor"], key='23')
                                                                                                        if (inp23 != "Select an Option"):
                                                                                                            lis.append(inp23)

                                                                                                            st.subheader("Question 24")
                                                                                                            st.write(questions[23])
                                                                                                            inp24 = st.selectbox("",["Select an Option","cloud computing","networks",
                                                                                                                                    "hacking","Computer Architecture"
                                                                                                                                    "programming","parallel computing"
                                                                                                                                    "IOT","data engineering"
                                                                                                                                    "Software Engineering","Management"], key='24')
                                                                                                            if (inp24 != "Select an Option"):
                                                                                                                lis.append(inp24)

                                                                                                                st.subheader("Question 25")
                                                                                                                st.write(questions[24])
                                                                                                                inp25 = st.selectbox("",["Select an Option","system developer","Business process analyst",
                                                                                                                                        "developer","testing","security",
                                                                                                                                        "cloud computing"
                                                                                                                                        ], key='25')
                                                                                                                if (inp25 != "Select an Option"):
                                                                                                                    lis.append(inp25)

                                                                                                                    st.subheader("Question 26")
                                                                                                                    st.write(questions[25])
                                                                                                                    inp26 = st.selectbox("",["Select an Option","higherstudies","job"], key='26')
                                                                                                                    if (inp26 != "Select an Option"):
                                                                                                                        lis.append(inp26)

                                                                                                                        st.subheader("Question 27")
                                                                                                                        st.write(questions[26])
                                                                                                                        inp27 = st.selectbox("",["Select an Option","Web Services","SAaS services","Sales and Marketing",
                                                                                                                                                "Testing and Maintainance Services","product development",
                                                                                                                                                "BPA","Service Based","Product based",
                                                                                                                                                "Cloud Services","Finance"], key='27')
                                                                                                                        if (inp27 != "Select an Option"):
                                                                                                                            lis.append(inp27)

                                                                                                                            st.subheader("Question 28")
                                                                                                                            st.write(questions[27])
                                                                                                                            inp28 = st.selectbox("",["Select an Option","yes","no"], key='28')
                                                                                                                            if (inp28 != "Select an Option"):
                                                                                                                                lis.append(inp28)

                                                                                                                                st.subheader("Question 29")
                                                                                                                                st.write(questions[28])
                                                                                                                                inp29 = st.selectbox("",["Select an Option","yes","no"], key='29')
                                                                                                                                if (inp29 != "Select an Option"):
                                                                                                                                    lis.append(inp29)

                                                                                                                                    st.subheader("Question 30")
                                                                                                                                    st.write(questions[29])
                                                                                                                                    inp30 = st.text_input("", key='30')
                                                                                                                                    if (inp30 != ""):
                                                                                                                                        lis.append(inp30)

                                                                                                                                        st.subheader("Question 31")
                                                                                                                                        st.write(questions[30])
                                                                                                                                        inp31 = st.selectbox("",["Select an Option","salary","Work"], key='31')
                                                                                                                                        if (inp31 != "Select an Option"):
                                                                                                                                            lis.append(inp31)

                                                                                                                                            st.subheader("Question 32")
                                                                                                                                            st.write(questions[31])
                                                                                                                                            inp32 = st.selectbox("",["Select an Option","yes","no"], key='32')
                                                                                                                                            if (inp32 != "Select an Option"):
                                                                                                                                                lis.append(inp32)

                                                                                                                                                st.subheader("Question 33")
                                                                                                                                                st.write(questions[32])
                                                                                                                                                inp33 = st.selectbox("",["Select an Option","stubborn","gentle"], key='33')
                                                                                                                                                if (inp33 != "Select an Option"):
                                                                                                                                                    lis.append(inp33)

                                                                                                                                                    st.subheader("Question 34")
                                                                                                                                                    st.write(questions[33])
                                                                                                                                                    inp34 = st.selectbox("",["Select an Option","Management","Technical"], key='34')
                                                                                                                                                    if (inp34 != "Select an Option"):
                                                                                                                                                        lis.append(inp34)

                                                                                                                                                        st.subheader("Question 35")
                                                                                                                                                        st.write(questions[34])
                                                                                                                                                        inp35 = st.selectbox("",["Select an Option","salary","work"], key='35')
                                                                                                                                                        if (inp35 != "Select an Option"):
                                                                                                                                                            lis.append(inp35)

                                                                                                                                                            st.subheader("Question 36")
                                                                                                                                                            st.write(questions[35])
                                                                                                                                                            inp36 = st.selectbox("",["Select an Option","hard worker","smart worker"], key='36')
                                                                                                                                                            if (inp36 != "Select an Option"):
                                                                                                                                                                lis.append(inp36)

                                                                                                                                                                st.subheader("Question 37")
                                                                                                                                                                st.write(questions[36])
                                                                                                                                                                inp37 = st.selectbox("",["Select an Option","yes","no"], key='37')
                                                                                                                                                                if (inp37 != "Select an Option"):
                                                                                                                                                                    lis.append(inp37)

                                                                                                                                                                    st.subheader("Question 38")
                                                                                                                                                                    st.write(questions[37])
                                                                                                                                                                    inp38 = st.selectbox("",["Select an Option","yes","no"], key='38')
                                                                                                                                                                    if (inp38 != "Select an Option"):
                                                                                                                                                                        lis.append(inp38)

                                                                                                                                                                        print(lis)

                                                                                                                                                                        st.success("Test Completed")                                                                                                                                                                        
                                                                                                                                                                        st.title("RESULTS:")
                                                                                                                                                                        result=output(lis)
                                                                                                                                                                        print(result)


                                                                                                                                                                        df = pd.read_csv(r'Results.csv', encoding= 'windows-1252')

                                                                                                                                                                        professions = {"Systems Security Administrator":1,
                                                                                                                                                                                    "Business Systems Analyst":2,
                                                                                                                                                                                    "Software Systems Engineer":3,
                                                                                                                                                                                    "Database Developer":4,
                                                                                                                                                                                    "Business Intelligence Analyst":5,
                                                                                                                                                                                    "CRM Technical Developer":6,
                                                                                                                                                                                    "Mobile Applications Developer":7,
                                                                                                                                                                                    "UX Designer":8,
                                                                                                                                                                                    "Quality Assurance Associate":9,
                                                                                                                                                                                    "Web Developer":10,
                                                                                                                                                                                    "Network Security Administrator":11}
                                                                                                                                                                        
                                                                                                                                                                        st.subheader("Recommended profession")
                                                                                                                                                                        st.subheader(result[0])
                                                                                                                                                                        st.write(df['Information'][professions[result[0]]-1])
                                                                                                                                                                        st.subheader('Average Monthly Income:')
                                                                                                                                                                        st.write("Rs. " + str(df['Income'][professions[result[0]]-1]))
                                                                                                                                                                        st.subheader("More About Profession:")
                                                                                                                                                                        st.write(df['more'][professions[result[0]]-1])
                                                                

                                                                                                                                                                        st.header('More information on the professions')
                                                                                                                                                                        # We'll be using a csv file for that
                                                                                                                   
                                                                                                                                                                        for i in range(0,5):
                                                                                                                                                                            st.subheader(df['Occupation'][i])
                                                                                                                                                                            st.write(df['Information'][i])

                                                                                                                                                                        st.header('Monthly Income')
                                                                                                                                                                        # We'll be using a csv file for that
                                                                                                                                                                        for i in range(0,5):
                                                                                                                                                                            st.subheader(df['Occupation'][i])
                                                                                                                                                                            st.write("Rs. " + str(df['Income'][i]))
                                                                                                                                                                                                                                                                                                                                         

if __name__=="__main__":
    main()