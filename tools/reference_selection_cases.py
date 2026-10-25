"""Singleton residual-sign boundaries and multi-predictor search controls."""
from reference_categorical_linear_cases import make as linear_case, data_files


def make(name, kind="categorical", role="both", groups=4, positive=12, size=24,
         *, missing=False, weights=False, competitor=False, constant=False, interactions=True,
         reverse=False):
    features=[("group",kind,role)]
    if competitor or constant:features.append(("other","categorical","split"))
    features.append(("x","numeric","fit"))
    rows=[];targets=[];ws=[]
    for g in range(groups):
        for x in range(size):
            level=(None if missing and g==groups-1 else "abcd"[g]) if kind=="categorical" else g
            row=[level]
            if competitor or constant:row.append("constant" if constant else "uv"[x%2])
            row.append(x);rows.append(row)
            h=size-positive if x>=size-positive else -positive
            targets.append(h*(1 if g<groups//2 else -1))
            ws.append(1+g%2)
    if reverse:
        rows.reverse();targets.reverse();ws.reverse()
    value=linear_case(name,features,rows,targets,weights=ws if weights else None)
    value["name"]="selection-"+name
    value["options"]["interactions"]=interactions
    value["expected_root_association"]="zero" if positive*2==size else "positive"
    value["expected_root_leaves"]=1 if positive*2==size else 2
    return value


def cases():
    result={}
    def add(value):result[value["name"]]=value
    for kind,role in [("categorical","both"),("categorical","split"),("numeric","both"),("numeric","split")]:
        for groups in [2,4]:
            for positive in [12,11]:
                add(make(f"{kind}-{role}-g{groups}-k{positive}",kind,role,groups,positive))
        add(make(f"{kind}-{role}-strong",kind,role,positive=1))
    for kind in ["categorical","numeric"]:
        add(make(kind+"-near-one",kind,"split",positive=119,size=240))
    for positive in [12,11]:
        for flag in ["missing","weights","constant","reverse"]:
            add(make(f"{flag}-k{positive}",positive=positive,**{flag:True}))
        add(make(f"no-interactions-k{positive}",positive=positive,interactions=False))
    # A positive but nonsignificant candidate must survive a zero-score rival
    # and Bonferroni screening; it still reaches the ordinary split search.
    for on in [False,True]:
        add(make("competitor-"+("on" if on else "off"),positive=11,competitor=True,interactions=on))
    for kind in ["categorical","numeric"]:
        rows=[];targets=[]
        for a in range(2):
            for b in range(2):
                for x in range(24):
                    rows.append(["ab"[a] if kind=="categorical" else a,"ab"[b] if kind=="categorical" else b,x])
                    targets.append(3 if a!=b else -3)
        value=linear_case("interaction-"+kind,[("a",kind,"split"),("b",kind,"split"),("x","numeric","fit")],rows,targets,depth=2)
        value["name"]="selection-interaction-"+kind
        value["expected_root_association"]="interaction";value["expected_root_leaves"]=4
        add(value)
    return result
