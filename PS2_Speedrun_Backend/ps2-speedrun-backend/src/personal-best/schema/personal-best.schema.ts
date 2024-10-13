import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type PersonalBestDocument = PersonalBest & Document;

@Schema()
export class PersonalBest {
  @Prop({ required: true })
  userId: string;

  @Prop({ required: true })
  personalBest: number;
}

export const PersonalBestSchema = SchemaFactory.createForClass(PersonalBest);
