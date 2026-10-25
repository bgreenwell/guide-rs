"""Numeric multiple-linear fallback boundaries and prediction limits."""
from reference_cases import case, data_files as base_files


def make(name, rows, targets, *, weights=None, split=False, depth=1, mode="none", pruning=None,
         expected_fit="linear"):
    p = len(rows[0])
    names = ["g", "z"] if split else [f"x{j}" for j in range(p)]
    value = case(name, False, [(n,"numeric") for n in names], rows, targets, weights=weights, depth=depth)
    for j,f in enumerate(value["dataset"]["schema"]): f["role"] = "split" if split and j==0 else "fit"
    value["options"]["node_model"] = {"kind":"linear"}
    if mode != "none": value["options"]["prediction_truncation"] = mode
    if pruning: value["options"]["pruning"] = pruning
    if depth>1:value["options"]["min_samples_leaf"] = 10
    value["expected_fit"] = expected_fit
    return value


def cases():
    result = {}
    specifications = [("complete",p,n) for p in [1,2,3,4,7] for n in [p+6,p+7]]
    specifications += [(kind,2,n) for kind in ["constant","collinear","weighted"] for n in [8,9]]
    specifications += [("zero-weight",2,n) for n in [9,10]]
    specifications += [("missing",1,n) for n in [4,5]]
    specifications += [("two-missing",2,n) for n in [6,7]]
    specifications += [("two-missing",4,n) for n in [8,9]]
    specifications += [("three-missing",7,n) for n in [12,13]]
    specifications += [("all-missing",2,n) for n in [5,6]]
    specifications += [(kind,2,n) for kind in ["excluded-missing","zero-weight-missing"] for n in [6,7]]
    for kind,p,n in specifications:
        rows=[];targets=[]
        for i in range(n):
            x=([i,(3*i)%7,(5*i)%11]+[(i*(2*j+1))%(2*j+7) for j in range(3,p)])[:p]
            if kind=="constant":x[-1]=1
            if kind=="collinear":x[-1]=2*x[0]
            if kind=="all-missing":x[-1]=None
            if kind=="missing" and i%5==0:x[-1]=None
            if kind in ["two-missing","three-missing"]:
                for j in range(2 if kind=="two-missing" else 3):
                    if i%(5+j)==j:x[j]=None
            rows.append(x);targets.append(3+2*i-(3*i)%7+(i%3-1))
        weights = [1+i%4 for i in range(n)] if kind in ["weighted","zero-weight"] else None
        if kind=="zero-weight":weights[0]=0
        if kind in ["excluded-missing","zero-weight-missing"]:
            rows.append([None]*p);targets.append(None if kind=="excluded-missing" else 999)
            if kind=="zero-weight-missing":weights=[1]*n+[0]
        active=sum(y is not None and (weights is None or weights[i]>0) for i,y in enumerate(targets))
        flags=sum(any(r[j] is None for r in rows) for j in range(p))
        minimum=p+7 if flags==0 else p+flags+3
        value=make(f"linear-policy-{kind}-p{p}-n{n}",rows,targets,weights=weights,
                   expected_fit="constant" if active<minimum else "linear")
        value["nominal_minimum"]=minimum
        result[value["name"]]=value
    value=make("linear-policy-global-flags",[[i//6,None if i==6 else i%6] for i in range(12)],
               [20*(i//6)+2*(i%6) for i in range(12)],split=True)
    result[value["name"]]=value
    for tree in [False,True]:
        for mode in ["none","node_range","node_range_expanded","global_range"]:
            rows=([[i//24,i%24] for i in range(48)]+[[0,100],[1,-100],[0,None],[1,None]]) if tree else ([[i] for i in range(20)]+[[-10],[30],[None]])
            targets=([3+2*(i%24)+100*(i//24) for i in range(48)]+[None]*4) if tree else ([3+2*i for i in range(20)]+[None]*3)
            value=make(f'linear-limit-{"tree" if tree else "root"}-{mode}',rows,targets,split=tree,mode=mode)
            result[value["name"]]=value
    for mode in ["node_range","global_range"]:
        value=make(f"linear-limit-loocv-{mode}",[[i//24,i%24] for i in range(48)],
                   [3+2*(i%24)+100*(i//24) for i in range(48)],split=True,mode=mode,
                   pruning={"kind":"cross_validation","folds":48,"standard_errors":0.25})
        result[value["name"]]=value
    value=make("linear-limit-training-quadratic",[[i] for i in range(20)]+[[-10],[30]],
               [i*i for i in range(20)]+[None,None],mode="node_range")
    result[value["name"]]=value
    for mode in ["none","node_range"]:
        value=make(f"linear-limit-fixed-{mode}",[[i,i%16] for i in range(64)],
                   [(i%16)**2+[0,100,400,520][i//16] for i in range(64)],split=True,depth=2,mode=mode,
                   pruning={"kind":"fixed_leaves","max_leaves":3})
        result[value["name"]]=value
    value=make("linear-limit-zero-weight-outlier",[[i] for i in range(20)]+[[100],[-10],[30]],
               [3+2*i for i in range(20)]+[10000,None,None],weights=[1]*20+[0,1,1],mode="global_range")
    result[value["name"]]=value
    return result


def data_files(value):
    files=base_files(value);schema=value["dataset"]["schema"];codes={"fit":"f","both":"n","split":"s"}
    files["data.dsc"]="data.txt\nNA\n2\n"+"".join(f'{j} {f["name"]} {codes[f["role"]]}\n' for j,f in enumerate(schema,1))+f"{len(schema)+1} y d\n"
    if value["dataset"]["weights"]:files["data.dsc"]+=f"{len(schema)+2} weight w\n"
    return files
