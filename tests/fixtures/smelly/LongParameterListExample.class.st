"I take too many arguments."
Class {
	#name : 'LongParameterListExample',
	#superclass : 'Object',
	#package : 'Smelly-Core'
}

{ #category : 'instance creation' }
LongParameterListExample >> withA: a b: b c: c d: d e: e [
	^ Array with: a with: b with: c with: d
]
