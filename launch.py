from _gsecrets import *

def start_vigil():
    print (f"""
        

        +------------------------------------+
          VIGIL 
          real time weapons detection system

          v{g_version}.20241023
          Gregory Roberts + John Heilman
        +------------------------------------+

        """)

    # user input of objects to detect
    searchtext = input("\nWelcome to VIGIL!\n\nType object classes to detect, separated by periods. \n(hit <enter> for default of 'person. face. gun.')\n>>> ")
    if searchtext == "":
        searchtext = "gun. person. face."

    labels = [value.strip() for value in searchtext.split(".")]

    return searchtext, labels
