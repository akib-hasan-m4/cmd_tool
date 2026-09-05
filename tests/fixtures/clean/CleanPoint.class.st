"I am a small, cohesive, well documented point."
Class {
	#name : 'CleanPoint',
	#superclass : 'Object',
	#instVars : [ 'x', 'y' ],
	#package : 'Clean-Core'
}

{ #category : 'accessing' }
CleanPoint >> x [
	^ x
]

{ #category : 'accessing' }
CleanPoint >> y [
	^ y
]

{ #category : 'arithmetic' }
CleanPoint >> + other [
	^ CleanPoint new setX: x + other x y: y + other y
]

{ #category : 'private' }
CleanPoint >> setX: aNumber y: anotherNumber [
	x := aNumber.
	y := anotherNumber
]
