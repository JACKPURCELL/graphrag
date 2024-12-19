import re
import json
import copy
# Sample text input
text = """
There are 10 software using the attack technique called LSASS Memory. Their names are as follows:
* SILENTTRINITY
* LaZagne
* Lslsass
* Impacket
* Pupy
* PoshC2
* Windows Credential Editor
* Mimikatz
* Empire
* PowerSploit

The following content describes information about softwares. The format of each software's information is as follows:
Software name: ***
Software description: ***
Platforms: ***
These respectively represent the name of the software, a detailed description, and the platforms on which it operates.

Software name: SILENTTRINITY
Software description: [SILENTTRINITY](https://attack.mitre.org/software/S0692) is an open source remote administration and post-exploitation framework primarily written in Python that includes stagers written in Powershell, C, and Boo. [SILENTTRINITY](https://attack.mitre.org/software/S0692) was used in a 2019 campaign against Croatian government agencies by unidentified cyber actors.(Citation: GitHub SILENTTRINITY March 2022)(Citation: Security Affairs SILENTTRINITY July 2019)
Platforms: Windows

Software name: LaZagne
Software description: [LaZagne](https://attack.mitre.org/software/S0349) is a post-exploitation, open-source tool used to recover stored passwords on a system. It has modules for Windows, Linux, and OSX, but is mainly focused on Windows systems. [LaZagne](https://attack.mitre.org/software/S0349) is publicly available on GitHub.(Citation: GitHub LaZagne Dec 2018)
Platforms: Linux, macOS, Windows

Software name: Lslsass
Software description: [Lslsass](https://attack.mitre.org/software/S0121) is a publicly-available tool that can dump active logon session password hashes from the lsass process. (Citation: Mandiant APT1)
Platforms: Windows

Software name: Impacket
Software description: [Impacket](https://attack.mitre.org/software/S0357) is an open source collection of modules written in Python for programmatically constructing and manipulating network protocols. [Impacket](https://attack.mitre.org/software/S0357) contains several tools for remote service execution, Kerberos manipulation, Windows credential dumping, packet sniffing, and relay attacks.(Citation: Impacket Tools)
Platforms: Linux, macOS, Windows

Software name: Pupy
Software description: [Pupy](https://attack.mitre.org/software/S0192) is an open source, cross-platform (Windows, Linux, OSX, Android) remote administration and post-exploitation tool. (Citation: GitHub Pupy) It is written in Python and can be generated as a payload in several different ways (Windows exe, Python file, PowerShell oneliner/file, Linux elf, APK, Rubber Ducky, etc.). (Citation: GitHub Pupy) [Pupy](https://attack.mitre.org/software/S0192) is publicly available on GitHub. (Citation: GitHub Pupy)
Platforms: Linux, Windows, macOS, Android

Software name: PoshC2
Software description: [PoshC2](https://attack.mitre.org/software/S0378) is an open source remote administration and post-exploitation framework that is publicly available on GitHub. The server-side components of the tool are primarily written in Python, while the implants are written in [PowerShell](https://attack.mitre.org/techniques/T1059/001). Although [PoshC2](https://attack.mitre.org/software/S0378) is primarily focused on Windows implantation, it does contain a basic Python dropper for Linux/macOS.(Citation: GitHub PoshC2)
Platforms: Windows, Linux, macOS

Software name: Windows Credential Editor
Software description: [Windows Credential Editor](https://attack.mitre.org/software/S0005) is a password dumping tool. (Citation: Amplia WCE)
Platforms: Windows

Software name: Mimikatz
Software description: [Mimikatz](https://attack.mitre.org/software/S0002) is a credential dumper capable of obtaining plaintext Windows account logins and passwords, along with many other features that make it useful for testing the security of networks. (Citation: Deply Mimikatz) (Citation: Adsecurity Mimikatz Guide)
Platforms: Windows

Software name: Empire
Software description: [Empire](https://attack.mitre.org/software/S0363) is an open source, cross-platform remote administration and post-exploitation framework that is publicly available on GitHub. While the tool itself is primarily written in Python, the post-exploitation agents are written in pure [PowerShell](https://attack.mitre.org/techniques/T1059/001) for Windows and Python for Linux/macOS. [Empire](https://attack.mitre.org/software/S0363) was one of five tools singled out by a joint report on public hacking tools being widely used by adversaries.(Citation: NCSC Joint Report Public Tools)(Citation: Github PowerShell Empire)(Citation: GitHub ATTACK Empire)
Platforms: Linux, macOS, Windows

Software name: PowerSploit
Software description: [PowerSploit](https://attack.mitre.org/software/S0194) is an open source, offensive security framework comprised of [PowerShell](https://attack.mitre.org/techniques/T1059/001) modules and scripts that perform a wide range of tasks related to penetration testing such as code execution, persistence, bypassing anti-virus, recon, and exfiltration. (Citation: GitHub PowerSploit May 2012) (Citation: PowerShellMagazine PowerSploit July 2014) (Citation: PowerSploit Documentation)
Platforms: Windows


There are 7 mitigation methods that can mitigating the attack technique called LSASS Memory. Their names are as follows:
* Operating System Configuration 
* Credential Access Protection 
* Privileged Process Integrity 
* Privileged Account Management 
* User Training 
* Behavior Prevention on Endpoint 
* Password Policies 

There are 10 software using the attack technique called LSASS Memory. Their names are as follows:
* SILENTTRINITY
* LaZagne
* Lslsass
* Impacket
* Pupy
* PoshC2
* Windows Credential Editor
* Mimikatz
* Empire
* PowerSploit

The following content describes information about mitigation methods. The format of each mitigation method's information is as follows:
Mitigation method name: ***
Mitigation method description: ***
These respectively represent the name of the mitigation method and a detailed description of how it works.

Mitigation method name: Operating System Configuration
Mitigation method description: Make configuration changes related to the operating system or a common feature of the operating system that result in system hardening against techniques.

Mitigation method name: Credential Access Protection
Mitigation method description: Use capabilities to prevent successful credential access by adversaries; including blocking forms of credential dumping.

Mitigation method name: Privileged Process Integrity
Mitigation method description: Protect processes with high privileges that can be used to interact with critical system components through use of protected process light, anti-process injection defenses, or other process integrity enforcement measures.

Mitigation method name: Privileged Account Management
Mitigation method description: Manage the creation, modification, use, and permissions associated to privileged accounts, including SYSTEM and root.

Mitigation method name: User Training
Mitigation method description: Train users to be aware of access or manipulation attempts by an adversary to reduce the risk of successful spearphishing, social engineering, and other techniques that involve user interaction.

Mitigation method name: Behavior Prevention on Endpoint
Mitigation method description: Use capabilities to prevent suspicious behavior patterns from occurring on endpoint systems. This could include suspicious process, file, API call, etc. behavior.

Mitigation method name: Password Policies
Mitigation method description: Set and enforce secure password policies for accounts.

"""


def extract_info(content):

    # Regular expression patterns
    software_pattern = r"software using the attack technique called (.+?)\. Their names are as follows:\n((?:\* .+\n)+)"
    mitigation_pattern = r"mitigation methods that can mitigating the attack technique called .+?\. Their names are as follows:\n((?:\* .+\n)+)"

    # Extract software names and attack technique
    software_match = re.search(software_pattern, content)
    attack_technique = software_match.group(1).strip()
    software_names = [name.strip('* ').strip() for name in software_match.group(2).strip().split('\n')]

    # Extract mitigation methods
    mitigation_match = re.search(mitigation_pattern, content)
    mitigation_methods = [method.strip('* ').strip() for method in mitigation_match.group(1).strip().split('\n')]

    # Template for the question and answer as a dictionary
    template = {
        "question": "How to mitigate the malicious software {SoftwareName}?",
        "gpt_answer_withoutdata": "xxx",
        "answer": "{SoftwareName} use attack techniques {AttackTechnique},which can be mitigated by using {MitigationMethods}.",
        "consistency": True,
        "root_nodes": "{SoftwareName}",
        "middle_node": "{AttackTechnique}",
        "leaf_nodes": "{MitigationMethodsList}",
        "chain_of_thoughts": [ ],
        "Template Relationship based on chain_of_thoughts": [
            "{root_node} use attack techniques {middle_node}.",
            "{middle_node} can be mitigated by using {leaf_node}."
        ],
        "Template Relationship between root and middle node": [
            "{root_node} use attack techniques {middle_node}."
        ],
        "Template Relationship between middle and leaf node": [
            "{middle_node} can be mitigated by using {leaf_node}."
        ],
        "Template Relationship between root and leaf node": [
            "{root_node} can be mitigated by using {leaf_node}."
        ],
        "knowledge_graph": []
    }

    # Generate the JSON for each software
    mitigation_methods_str = ', '.join(mitigation_methods)
    knowledge_graph_entries = [[attack_technique, method, "mitigate"] for method in mitigation_methods]
    return_jsons = []
    
    for software in software_names:
        # Fill in the template with actual values
        filled_template = {}
        new_template = copy.deepcopy(template)
        filled_template = {key: value.format(
            SoftwareName=software,
            AttackTechnique=attack_technique,
            MitigationMethods=mitigation_methods_str,
            MitigationMethodsList=mitigation_methods  # Directly use the list
           # Convert list of lists to JSON
        ) if isinstance(value, str) else value for key, value in new_template.items()}
        filled_template["leaf_nodes"] = mitigation_methods
        # filled_template["knowledge_graph"] = [],
        filled_template["knowledge_graph"].append([software, attack_technique, "attack techniques"])
        filled_template["knowledge_graph"].extend(knowledge_graph_entries)
        filled_template["chain_of_thoughts"].append([ f"{software} use attack techniques f{attack_technique}."]),
        filled_template["chain_of_thoughts"].extend(
            [f"{attack_technique} can be mitigated by using {method}."] for method in mitigation_methods
        )

        # Append the filled template to the list
      
        output_json = {}
        output_json["questions"] = [filled_template]
        output_json["middle_node"] = filled_template["middle_node"]
        output_json["as_source"] = [[filled_template["middle_node"], leaf_node] for leaf_node in filled_template["leaf_nodes"]]
        output_json["as_target"] = [[filled_template["root_nodes"],filled_template["middle_node"]]]
        return_jsons.append(output_json)
    # for each_j in return_jsons:
        
    #     output_json["as_target"].append([each_j["root_nodes"], each_j["middle_node"]])
    

    
    return return_jsons

    
import os
# Define the directory containing the .txt files
directory = '/home/ljc/data/graphrag/alltest/new_corpus_1207/cyber_v3_tobeuse/input'
save_jsons = []
# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Open and read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            save_jsons.extend(extract_info(content))
            
            
with open('/home/ljc/data/graphrag/alltest/new_corpus_1207/cyber_v3_tobeuse/input/question_multi_v3.json', 'w') as f:
    json.dump(save_jsons, f, indent=4)
            
