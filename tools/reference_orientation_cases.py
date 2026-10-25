"""Linear categorical partition orientation and actual held-out routes."""
from reference_categorical_linear_cases import make as linear_case


def make(name,levels,plus=(0,1),role="both",*,weighted=False,reverse=False,pruning=None):
    rows=[];targets=[];weights=[]
    for g,level in enumerate(levels):
        for x in range(24):
            rows.append([level,x]);targets.append((13 if x>=13 else -11)*(1 if g in plus else -1))
            weights.append(1+g%2)
    if reverse:rows.reverse();targets.reverse();weights.reverse()
    if not pruning:
        queries=["!future","zfuture",None]+[v for v in levels if v is not None]
        rows += [[v,23] for v in queries];targets += [None]*len(queries);weights += [1]*len(queries)
    value=linear_case(name,[("group","categorical",role),("x","numeric","fit")],rows,targets,
                      weights=weights if weighted else None,pruning=pruning)
    value["name"]="orientation-"+name
    return value


def cases():
    result={}
    def add(value):result[value["name"]]=value
    for name,levels in [
        ("complete",["a","b","c","d"]),
        ("missing-lower",["a","b","c",None]),
        ("missing-upper",["A","b","c",None]),
        ("numeric-labels",["2","10","01",None]),
        ("reordered",["zeta","beta","alpha","delta"]),
    ]:
        for second in [1,2,3]:
            for role in ["both","split"]:
                add(make(f"{name}-p{second}-{role}",levels,(0,second),role))
    for name,label in [("lower","a"),("upper","A")]:
        for role in ["both","split"]:
            add(make(f"missing-only-{name}-{role}",[label,None],(0,),role))
    add(make("weighted",["a","b","c",None],weighted=True))
    add(make("reversed",["a","b","c",None],reverse=True))
    add(make("loocv",["a","b","c",None],pruning={"kind":"cross_validation","folds":96,"standard_errors":0.25}))
    # The first global level (NA) is absent from the first child. Local
    # orientation must use that child's first observed level instead.
    for role in ["both","split"]:
        rows=[];targets=[]
        for outer,levels in [(0,["a","b","c","d"]),(1,["A","b","c",None])]:
            for g,level in enumerate(levels):
                for x in range(24):
                    rows.append([outer,level,x]);targets.append(1000*outer+(13 if x>=13 else -11)*(1 if g in (0,3) else -1))
        rows += [[outer,label,23] for outer in [0,1] for label in ["!future","zfuture",None]]
        targets += [None]*6
        value=linear_case("local-"+role,[("outer","numeric","split"),("group","categorical",role),("x","numeric","fit")],rows,targets,depth=2)
        value["name"]="orientation-local-"+role;add(value)
    return result
