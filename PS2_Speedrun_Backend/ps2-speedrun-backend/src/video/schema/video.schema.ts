import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

export type VideoDocument = Video & Document;

@Schema()
export class Video {
  @Prop({ required: true })
  userId: string;

  @Prop({ required: true })
  videoUrl: string;

  @Prop({ required: true })
  fileSize: number;

  @Prop({ default: Date.now })
  uploadTime: Date;
}

export const VideoSchema = SchemaFactory.createForClass(Video);
