"I reach through several objects to get what I want."
Class {
	#name : 'MessageChainExample',
	#superclass : 'Object',
	#instVars : [ 'order' ],
	#package : 'Smelly-Core'
}

{ #category : 'accessing' }
MessageChainExample >> streetName [
	^ order customer address street name asUppercase
]
