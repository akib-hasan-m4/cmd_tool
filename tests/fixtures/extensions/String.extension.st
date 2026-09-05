Extension { #name : 'String' }

{ #category : '*MyPackage' }
String >> myShout [
	^ self asUppercase
]

{ #category : '*MyPackage' }
String >> myWhisper [
	^ self asLowercase
]
