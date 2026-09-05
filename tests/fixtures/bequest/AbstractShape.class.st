"I am the superclass that provides a real implementation."
Class {
	#name : 'AbstractShape',
	#superclass : 'Object',
	#instVars : [ 'origin' ],
	#package : 'Bequest-Core'
}

{ #category : 'drawing' }
AbstractShape >> draw [
	^ origin
]

{ #category : 'drawing' }
AbstractShape >> area [
	^ 0
]
