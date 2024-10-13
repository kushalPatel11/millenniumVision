import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { VideoModule } from './video/video.module';
import { PersonalBestModule } from './personal-best/personal-best.module';

@Module({
  imports: [
    MongooseModule.forRoot('mongodb://localhost:27017/ps2SpeedRun'), // MongoDB connection
    VideoModule,
    PersonalBestModule,
  ],
})
export class AppModule {}
