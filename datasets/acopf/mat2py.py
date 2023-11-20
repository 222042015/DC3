def convert_matlab_to_python(matlab_code):
    # Extract data between [ and ]
    def extract_data(s):
        return [line.strip().strip(';').split() for line in s.split('[')[1].split(']')[0].strip().split('\n') if line.strip()]

    # Extract data for specific keyword
    def get_data(keyword):
        start = matlab_code.find(f"{keyword} = [")
        end = matlab_code.find("];", start) + 2
        return extract_data(matlab_code[start:end])

    lines = matlab_code.split('\n')
    lines = [line.split('%', 1)[0] for line in lines]
    matlab_code = '\n'.join(lines)

    ppc = {"version": '2'}

    # Extract system MVA base
    baseMVA_line = next((line for line in matlab_code.splitlines() if 'mpc.baseMVA' in line), None)
    ppc["baseMVA"] = float(baseMVA_line.split('=')[1].strip(';'))

    # Extract bus data
    bus_data = get_data("mpc.bus")
    nbus = len(bus_data)
    ppc["bus"] = 'array(['
    for bus in bus_data:
        ppc["bus"] += f"\n\t[{', '.join(bus)}],"
    ppc["bus"] += "\n])"

    # Extract generator data
    gen_data = get_data("mpc.gen")
    ppc["gen"] = 'array(['
    for gen in gen_data:
        ppc["gen"] += f"\n\t[{', '.join(gen)}],"
    ppc["gen"] += "\n])"

    # Extract branch data
    branch_data = get_data("mpc.branch")
    ppc["branch"] = 'array(['
    for branch in branch_data:
        ppc["branch"] += f"\n\t[{', '.join(branch)}],"
    ppc["branch"] += "\n])"

    gencost_data = get_data("mpc.gencost")
    ppc["gencost"]  = 'array(['
    for gencost in gencost_data:
        ppc["gencost"] += f"\n\t[{', '.join(gencost)}],"
    ppc["gencost"] += "\n])"
    # Construct the Python function
    python_code = f"""from numpy import array

def case{nbus}():
    \"\"\"Power flow data for IEEE 14 bus test case.
    Please see L{{caseformat}} for details on the case file format.

    @return: Power flow data for IEEE 14 bus test case.
    \"\"\"
    ppc = {{"version": '2'}}

    ## system MVA base
    ppc["baseMVA"] = {ppc["baseMVA"]}

    ## bus data
    ppc["bus"] = {ppc["bus"]}

    ## generator data
    ppc["gen"] = {ppc["gen"]}

    ## branch data
    ppc["branch"] = {ppc["branch"]}
    
    ## gencost data
    ppc["gencost"]={ppc["gencost"]}

    return ppc"""

    return python_code, nbus

if __name__ == '__main__':
    # read in the content from the case39.m file
    nbus = 1354
    case_dict = {"57": "pglib_opf_case57_ieee.m", 
                 "39": "pglib_opf_case39_epri.m",
                 "118": "pglib_opf_case118_ieee.m",
                 "1354": "pglib_opf_case1354_pegase.m",
                 "300": "pglib_opf_case300_ieee.m"}

    with open("./case/" + case_dict[str(nbus)], 'r') as f:
        matlab_code = f.read()

    python_code, nbus = convert_matlab_to_python(matlab_code)
output_file = './case/case' + str(nbus) + '.py'
with open(output_file, 'w') as f:
    f.write(python_code)
