import re
def ToVariableName(nameStr, withPrefix=True, Overlength=True) :
    if len(nameStr)>50 and Overlength==False:
        splitSymLst = [':', '\.', '/', '-', ';', '_']
        shortName = 'TS_'
        _splitted = False
        for i in range(len(nameStr)):
            if nameStr[i] in splitSymLst:
                if _splitted is False:
                    shortName += '_'
                shortName += nameStr[i+1]
                _splitted = True
            if _splitted is False:
                shortName += nameStr[i]
            elif nameStr[i].isdigit():
                shortName += nameStr[i]
        return shortName
    elif nameStr == '':
        return nameStr
    else:
        s = re.sub(':', '_{:02x}_'.format(ord(':')), nameStr)
        s = re.sub('\.', '_{:02x}_'.format(ord('.')), s)
        s = re.sub('/', '_{:02x}_'.format(ord('/')), s)
        s = re.sub('-', '_{:02x}_'.format(ord('-')), s)
        s = re.sub(';', '_{:02x}_'.format(ord(';')), s)
        if withPrefix:
            s = "Tensor_"+s if s[0].isdigit() else s
            s = "Tensor"+s if s[0]=='_' else s
        return s
