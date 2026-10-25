"""Multiple-linear SSE categorical designs: global levels, local aliases and guards."""
from reference_cases import case, data_files as base_files


def make(name, features, rows, targets, *, weights=None, depth=1, mode="none", pruning=None):
    value=case("categorical-linear-"+name,False,[(n,k) for n,k,_ in features],rows,targets,weights=weights,depth=depth)
    for f,(_,_,role) in zip(value["dataset"]["schema"],features):f["role"]=role
    value["options"]["node_model"]={"kind":"linear"}
    if mode!="none":value["options"]["prediction_truncation"]=mode
    if pruning:value["options"]["pruning"]=pruning
    value["expected_fit"]="linear"
    return value


def cases():
    result={};cat=[("group","categorical","fit")]
    def add(value):result[value["name"]]=value
    for kind,k in [("complete",2),("complete",3),("complete",4),("missing",3),
                   ("excluded-level",3),("excluded-missing",3),("mixed",3),
                   ("numeric-missing",3),("both-missing",3),("zero-level",2),("zero-missing",2)]:
        numeric=kind in ("mixed","numeric-missing","both-missing")
        d=k-1+(kind in ("missing","both-missing","excluded-level","excluded-missing","zero-level","zero-missing"))
        p=d+numeric;m=kind in ("numeric-missing","both-missing")
        minimum=p+7 if not m else p+4
        for n in [minimum-1,minimum]:
            rows=[];ys=[]
            for i in range(n):
                group=None if kind in ("missing","both-missing") and i%7==0 else "abcd"[i%k]
                x=None if m and i%5==0 else (i*3)%7
                rows.append([group]+([x] if numeric else []))
                ys.append((20 if group is None else 2+6*(i%k))+(2*(0 if x is None else x) if numeric else 0)+(i%2)*0.5)
            weights=None
            if kind.startswith("excluded") or kind.startswith("zero"):
                rows.append([None if "missing" in kind else "z"])
                ys.append(999 if kind.startswith("zero") else None)
                if kind.startswith("zero"):weights=[1]*n+[0]
            value=make(f"{kind}-k{k}-n{n}",cat+([("x","numeric","fit")] if numeric else []),rows,ys,weights=weights)
            value["nominal_minimum"]=minimum
            value["expected_fit"]="constant" if n<minimum else "linear"
            add(value)
    for name,levels,extra in [
        ("reference",["a","b","c"],["z"]),
        ("inactive-reference",["a","b","c"],["z",None]),
        ("observed-missing",["a","b","c",None],["z"]),
        ("numeric-labels",["2","10","01"],["z",None]),
        ("case-order",["a","B","A"],["z",None]),
        ("reverse-order",["zeta","beta","alpha"],["unknown",None]),
        ("all-missing",[None],["a","b"]),
        ("constant",["a"],[]),
    ]:
        rows=[[levels[i%len(levels)]] for i in range(36)]+[[x] for x in extra]
        ys=[2+6*(i%len(levels)) for i in range(36)]+[None]*len(extra)
        add(make(name,cat,rows,ys))
    rows=[["abc"[i%3],"uv"[(i//3)%2]] for i in range(36)]
    ys=[2+6*(i%3)+3*((i//3)%2)+(i%2)*0.25 for i in range(36)]
    add(make("two-weighted",cat+[("other","categorical","fit")],rows,ys,weights=[1+i%4 for i in range(36)]))
    add(make("duplicate",cat+[("other","categorical","fit")],[["abc"[i%3]]*2 for i in range(36)],
             [2+6*(i%3) for i in range(36)]))
    for missing in [False,True]:
        rows=[];ys=[]
        for i in range(48):
            g=i//24;group=("ab" if g==0 else "bc")[i%2];x=None if missing and i==0 else i%6
            rows.append([g,group,x]);ys.append(20*g+{"a":2,"b":8,"c":14}[group]+3*(i%6))
        rows += [[0,"z",10],[1,"z",10]];ys += [None,None]
        add(make("local-reference"+("-missing" if missing else ""),[("g","numeric","split")]+cat+[("x","numeric","fit")],rows,ys))
    rows=[["abcd"[i//24],i%24] for i in range(96)]
    ys=[(2 if i//24<2 else -3)*(i%24)+6*(i//24) for i in range(96)]
    add(make("split-and-fit",[("group","categorical","both"),("x","numeric","fit")],rows,ys))
    ys=[(1 if i//24<2 else -1)*(100 if i%24==23 else 0)+6*(i//24) for i in range(96)]
    add(make("split-and-fit-skewed",[("group","categorical","both"),("x","numeric","fit")],rows,ys))
    for mode in ["none","node_range","node_range_expanded","global_range"]:
        rows=[["abc"[i%3],i//3] for i in range(36)]+[["z",30],[None,-10]]
        ys=[2+6*(i%3)+3*(i//3) for i in range(36)]+[None,None]
        add(make("limits-"+mode,cat+[("x","numeric","fit")],rows,ys,mode=mode))
    rows=[[i//24,"abc"[i%3],i%6] for i in range(48)]
    ys=[20*(i//24)+2+6*(i%3)+3*(i%6)+(i%2)*0.25 for i in range(48)]
    add(make("loocv",[("g","numeric","split")]+cat+[("x","numeric","fit")],rows,ys,
             pruning={"kind":"cross_validation","folds":48,"standard_errors":0.25}))
    return result


def data_files(value):
    files=base_files(value);schema=value["dataset"]["schema"]
    codes={"categorical":{"fit":"i","both":"b","split":"c"},"numeric":{"fit":"f","both":"n","split":"s"}}
    files["data.dsc"]="data.txt\nNA\n2\n"+"".join(f'{j} {f["name"]} {codes[f["feature_type"]["kind"]][f["role"]]}\n' for j,f in enumerate(schema,1))+f"{len(schema)+1} y d\n"
    if value["dataset"]["weights"]:files["data.dsc"]+=f"{len(schema)+2} weight w\n"
    return files
